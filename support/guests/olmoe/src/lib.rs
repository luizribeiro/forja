//! OLMoE-1B-7B-0924 bf16 inference engine.

#[cfg(target_family = "wasm")]
use forja_sdk::nn::blocks::{ChunkedPrefill, DEFAULT_PREFILL_CHUNK, DecodeSelection, DecodeState};
use forja_sdk::{
    DType, Dim, Engine, EngineInfo, Load, Param, Result, StepInput, StepOutput, Tensor, Weights,
    bf16, export_engine,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig,
        blocks::{KvCache, Taps, cached_attention_with},
        moe_combine_with, moe_router_with, stack_expert_weights,
    },
    target::metal::{Variant, VariantChoice, VariantRule},
};

/// Vocabulary size of OLMoE-1B-7B-0924.
pub const VOCAB: u32 = 50_304;
/// Hidden width of OLMoE-1B-7B-0924.
pub const HIDDEN: u32 = 2_048;
/// Number of decoder layers.
pub const LAYERS: usize = 16;
/// Largest context accepted by this engine.
pub const MAX_CONTEXT: u32 = 4_096;

const HEADS: u32 = 16;
const HEAD_DIM: u32 = 128;
const INTERMEDIATE: u32 = 1_024;
/// Number of routed experts.
pub const EXPERTS: u32 = 64;
/// Experts selected for each token.
pub const TOP_K: u32 = 8;
const RMS_EPSILON: f32 = 1.0e-5;
const ROPE_THETA: f32 = 10_000.0;
const ATTENTION_SCALE: f32 = 0.088_388_35;
#[cfg(target_family = "wasm")]
const REPLAY_DECODE: bool = !cfg!(feature = "no-replay");

#[derive(Clone, Copy)]
struct Config;

type SequenceOutput = (Tensor<bf16>, Vec<Tensor<f32>>, Vec<Tensor<f32>>);

enum AttentionVariant {
    Fixed(Variant),
    Rule(VariantRule),
}

impl AttentionVariant {
    fn choice(&self) -> VariantChoice<'_> {
        match self {
            Self::Fixed(variant) => variant.into(),
            Self::Rule(rule) => rule.into(),
        }
    }
}

#[cfg(target_family = "wasm")]
#[derive(Clone, Copy)]
struct PrefillVariant<'a> {
    start: u32,
    parameter: Option<&'a Param>,
}

fn variant(name: &str) -> Result<Variant> {
    Variant::new(name)
}

fn matmul_variant(
    dtype: DType,
    batch: u32,
    rows: u32,
    columns: u32,
    inner: u32,
    transposed_right: bool,
) -> Result<Variant> {
    if rows == 1 {
        return variant(if transposed_right {
            "matmul.gemv-transposed"
        } else {
            "matmul.gemv"
        });
    }
    let large = u64::from(batch)
        .checked_mul(u64::from(rows))
        .and_then(|elements| elements.checked_mul(u64::from(columns)))
        .ok_or_else(|| forja_sdk::Error::loading("matmul output size overflowed"))?
        >= 1 << 20;
    let half = matches!(dtype, DType::F16 | DType::BF16);
    let name = if half && large && u64::from(rows.max(columns)) * 2 > u64::from(inner) {
        "matmul.steel-64x64x16-1x2"
    } else if half && transposed_right {
        "matmul.steel-64x32x32-2x2"
    } else if half && large {
        "matmul.steel-32x64x16-1x2"
    } else if half {
        "matmul.steel-64x64x16-1x2"
    } else if !large && transposed_right {
        "matmul.steel-32x64x16-1x2"
    } else if !large {
        "matmul.steel-64x32x32-2x2"
    } else {
        "matmul.steel-64x64x16-2x2"
    };
    variant(name)
}

fn linear<T: forja_sdk::Element>(
    projection: &Linear<T>,
    input: &Tensor<T>,
    dtype: DType,
    rows: u32,
    inner: u32,
    columns: u32,
) -> Result<Tensor<T>> {
    let selection = matmul_variant(dtype, 1, rows, columns, inner, true)?;
    projection.forward_with(input, &selection)
}

fn attention_variant(
    sequence: u32,
    start: u32,
    parameter: Option<&Param>,
) -> Result<AttentionVariant> {
    if let Some(parameter) = parameter {
        let arms = if sequence == 1 {
            vec![
                (0..=510, variant("sdpa.decomposed")?),
                (511..=1022, variant("sdpa.vector-single-pass")?),
                (1023..=MAX_CONTEXT - 1, variant("sdpa.vector-two-pass")?),
            ]
        } else if sequence >= 512 {
            vec![
                (0..=0, variant("sdpa.steel")?),
                (1..=MAX_CONTEXT - sequence, variant("sdpa.decomposed")?),
            ]
        } else {
            vec![(0..=MAX_CONTEXT - sequence, variant("sdpa.decomposed")?)]
        };
        return Ok(AttentionVariant::Rule(VariantRule::new(parameter, arms)?));
    }
    let key_length = start
        .checked_add(sequence)
        .ok_or_else(|| forja_sdk::Error::loading("attention cache length overflowed"))?;
    let name = if sequence == 1 {
        match key_length {
            ..512 => "sdpa.decomposed",
            512..1024 => "sdpa.vector-single-pass",
            _ => "sdpa.vector-two-pass",
        }
    } else if sequence >= 512 && key_length == sequence {
        "sdpa.steel"
    } else {
        "sdpa.decomposed"
    };
    Ok(AttentionVariant::Fixed(variant(name)?))
}

#[derive(Load)]
#[load(config = Config)]
struct Attention {
    #[load(prefix, config = LinearConfig::promoted_bf16(HIDDEN, HIDDEN))]
    q_proj: Linear<f32>,
    #[load(prefix, config = LinearConfig::promoted_bf16(HIDDEN, HIDDEN))]
    k_proj: Linear<f32>,
    #[load(prefix, config = LinearConfig::promoted_bf16(HIDDEN, HIDDEN))]
    v_proj: Linear<f32>,
    #[load(prefix, config = LinearConfig::promoted_bf16(HIDDEN, HIDDEN))]
    o_proj: Linear<f32>,
    #[load(prefix, config = RmsNormConfig::promoted_bf16(HIDDEN, RMS_EPSILON))]
    q_norm: RmsNorm<f32>,
    #[load(prefix, config = RmsNormConfig::promoted_bf16(HIDDEN, RMS_EPSILON))]
    k_norm: RmsNorm<f32>,
}

struct SparseMoe {
    gate: Linear<f32>,
    gate_proj: Tensor<bf16>,
    up_proj: Tensor<bf16>,
    down_proj: Tensor<bf16>,
}

impl Load<Config> for SparseMoe {
    fn load(weights: &Weights<'_>, _config: &Config) -> Result<Self> {
        let experts = weights.scoped("experts");
        Ok(Self {
            gate: Linear::load(
                &weights.scoped("gate"),
                &LinearConfig::promoted_bf16(HIDDEN, EXPERTS),
            )?,
            gate_proj: stack_expert_weights(&experts, EXPERTS, HIDDEN, INTERMEDIATE, "gate_proj")?,
            up_proj: stack_expert_weights(&experts, EXPERTS, HIDDEN, INTERMEDIATE, "up_proj")?,
            down_proj: stack_expert_weights(&experts, EXPERTS, INTERMEDIATE, HIDDEN, "down_proj")?,
        })
    }
}

impl SparseMoe {
    fn forward(&self, input: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let rows = input
            .shape()
            .first()
            .copied()
            .ok_or_else(|| forja_sdk::Error::loading("MoE input must have rank two"))?;
        let logits = linear(&self.gate, input, DType::F32, rows, HIDDEN, EXPERTS)?;
        let top_k = variant("top-k.single-k8")?;
        let (route_weights, indices) = moe_router_with(&logits, TOP_K, false, &top_k)?;
        let expert_input = input.to_dtype::<bf16>()?;
        let gate = expert_input.gather_matmul(&self.gate_proj, &indices)?;
        let up = expert_input.gather_matmul(&self.up_proj, &indices)?;
        let activated = gate.silu_mul(&up)?;
        let routed_rows = rows
            .checked_mul(TOP_K)
            .ok_or_else(|| forja_sdk::Error::loading("routed row count overflowed"))?;
        let output = activated
            .reshape(&[routed_rows, INTERMEDIATE])?
            .gather_matmul(&self.down_proj, &indices.reshape(&[routed_rows, 1])?)?
            .reshape(&[rows, TOP_K, HIDDEN])?;
        let combine = matmul_variant(DType::BF16, rows, HIDDEN, 1, TOP_K, false)?;
        Ok((
            moe_combine_with(&output, &route_weights.to_dtype()?, &combine)?.to_dtype()?,
            logits,
        ))
    }
}

#[derive(Load)]
#[load(config = Config)]
struct DecoderLayer {
    #[load(prefix, config = RmsNormConfig::promoted_bf16(HIDDEN, RMS_EPSILON))]
    input_layernorm: RmsNorm<f32>,
    #[load(prefix)]
    self_attn: Attention,
    #[load(prefix, config = RmsNormConfig::promoted_bf16(HIDDEN, RMS_EPSILON))]
    post_attention_layernorm: RmsNorm<f32>,
    #[load(prefix)]
    mlp: SparseMoe,
}

impl DecoderLayer {
    fn forward(
        &self,
        input: &Tensor<f32>,
        positions: &Tensor<u32>,
        cache: &mut KvCache<f32>,
        start: &Dim,
        end: &Dim,
        attention_variant: &AttentionVariant,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let sequence = positions.shape()[0];
        let normalized = self.input_layernorm.forward(input)?;
        let query_projection = linear(
            &self.self_attn.q_proj,
            &normalized,
            DType::F32,
            sequence,
            HIDDEN,
            HIDDEN,
        )?;
        let query = self
            .self_attn
            .q_norm
            .forward(&query_projection)?
            .reshape(&[sequence, HEADS, HEAD_DIM])?
            .rope(positions, ROPE_THETA)?;
        let key_projection = linear(
            &self.self_attn.k_proj,
            &normalized,
            DType::F32,
            sequence,
            HIDDEN,
            HIDDEN,
        )?;
        let key = self
            .self_attn
            .k_norm
            .forward(&key_projection)?
            .reshape(&[sequence, HEADS, HEAD_DIM])?
            .rope(positions, ROPE_THETA)?;
        let value = linear(
            &self.self_attn.v_proj,
            &normalized,
            DType::F32,
            sequence,
            HIDDEN,
            HIDDEN,
        )?
        .reshape(&[sequence, HEADS, HEAD_DIM])?;
        let attended = cached_attention_with(
            &query,
            &key,
            &value,
            cache,
            ATTENTION_SCALE,
            start,
            end,
            attention_variant.choice(),
        )?;
        let attention = linear(
            &self.self_attn.o_proj,
            &attended,
            DType::F32,
            sequence,
            HIDDEN,
            HIDDEN,
        )?;
        let hidden = (input + &attention)?;
        let normalized = self.post_attention_layernorm.forward(&hidden)?;
        let (projected, router_logits) = self.mlp.forward(&normalized)?;
        Ok(((&hidden + &projected)?, router_logits))
    }
}

#[derive(Load)]
#[load(config = Config)]
struct Model {
    #[load(prefix, config = EmbeddingConfig::promoted_bf16(VOCAB, HIDDEN))]
    embed_tokens: Embedding<f32>,
    #[load(prefix, count = LAYERS)]
    layers: Vec<DecoderLayer>,
    #[load(prefix, config = RmsNormConfig::promoted_bf16(HIDDEN, RMS_EPSILON))]
    norm: RmsNorm<f32>,
}

#[derive(Load)]
#[load(config = Config)]
struct OlmoeWeights {
    #[load(prefix)]
    model: Model,
    #[load(prefix, config = LinearConfig::new(HIDDEN, VOCAB))]
    lm_head: Linear<bf16>,
}

/// OLMoE-1B-7B-0924 with a fixed 4096-token KV cache.
pub struct Olmoe {
    weights: OlmoeWeights,
    caches: Vec<KvCache<f32>>,
    positions: Tensor<u32>,
    #[cfg(target_family = "wasm")]
    decode: DecodeState,
    #[cfg(target_family = "wasm")]
    prefill: Option<ChunkedPrefill<f32>>,
}

impl Olmoe {
    fn load_from_weights(weights: &Weights<'_>) -> Result<Self> {
        let weights = OlmoeWeights::load(weights, &Config)?;
        let caches = (0..LAYERS)
            .map(|_| KvCache::new(HEADS, MAX_CONTEXT, HEAD_DIM, 0.0))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights,
            caches,
            positions: Tensor::constant(&(0..MAX_CONTEXT).collect::<Vec<_>>(), &[MAX_CONTEXT])?,
            #[cfg(target_family = "wasm")]
            decode: DecodeState::new(MAX_CONTEXT)?,
            #[cfg(target_family = "wasm")]
            prefill: Some(ChunkedPrefill::new(MAX_CONTEXT, DEFAULT_PREFILL_CHUNK)?),
        })
    }

    fn forward(
        &mut self,
        tokens: &Tensor<u32>,
        sequence: u32,
        start: &Dim,
        end: &Dim,
        taps_enabled: bool,
        attention_variant: &AttentionVariant,
    ) -> Result<StepOutput> {
        let (logits, taps, router_logits) = self.forward_sequence(
            tokens,
            sequence,
            start,
            end,
            taps_enabled,
            attention_variant,
        )?;
        Ok(StepOutput {
            logits: Self::last_logits(
                &logits,
                sequence
                    .checked_sub(1)
                    .ok_or_else(|| forja_sdk::Error::loading("tokens cannot be empty"))?,
            )?,
            taps,
            router_logits,
        })
    }

    fn forward_sequence(
        &mut self,
        tokens: &Tensor<u32>,
        sequence: u32,
        start: &Dim,
        end: &Dim,
        taps_enabled: bool,
        attention_variant: &AttentionVariant,
    ) -> Result<SequenceOutput> {
        let positions = self.positions.narrow(0, start, sequence)?;
        let mut hidden = self.weights.model.embed_tokens.forward(tokens)?;
        let mut taps = Taps::new(taps_enabled, LAYERS);
        let mut routers = Vec::with_capacity(if taps_enabled { LAYERS } else { 0 });
        for index in 0..LAYERS {
            let (next, router) = self.weights.model.layers[index].forward(
                &hidden,
                &positions,
                &mut self.caches[index],
                start,
                end,
                attention_variant,
            )?;
            hidden = next;
            if taps.enabled() {
                routers.push(router.contiguous()?);
                if index + 1 < LAYERS {
                    taps.push(hidden.contiguous()?);
                }
            }
        }
        hidden = self.weights.model.norm.forward(&hidden)?;
        if taps.enabled() {
            taps.push(hidden.contiguous()?);
        }
        let logits = linear(
            &self.weights.lm_head,
            &hidden.to_dtype()?,
            DType::BF16,
            sequence,
            HIDDEN,
            VOCAB,
        )?;
        Ok((logits, taps.finish(), routers))
    }

    fn last_logits(logits: &Tensor<bf16>, last: impl Into<Dim>) -> Result<Tensor<f32>> {
        logits
            .narrow(0, last, 1)?
            .reshape(&[VOCAB])?
            .to_dtype::<f32>()?
            .contiguous()
    }

    #[cfg(target_family = "wasm")]
    fn forward_prefill_chunk(
        &mut self,
        tokens: &Tensor<u32>,
        sequence: u32,
        last: &Dim,
        start: &Dim,
        end: &Dim,
        selection: PrefillVariant<'_>,
    ) -> Result<Tensor<f32>> {
        let attention = attention_variant(sequence, selection.start, selection.parameter)?;
        let (logits, _, _) =
            self.forward_sequence(tokens, sequence, start, end, false, &attention)?;
        Self::last_logits(&logits, last)
    }
}

#[export_engine]
impl Engine for Olmoe {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: VOCAB,
            max_context: MAX_CONTEXT,
            tap_layers: (1..=16).collect(),
            router_layers: (1..=16).collect(),
        }
    }

    fn load(weights: &Weights<'_>, _config: forja_sdk::EngineLoadConfig) -> Result<Self> {
        Self::load_from_weights(weights)
    }

    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
        let [sequence] = input
            .tokens
            .shape()
            .try_into()
            .map_err(|_| forja_sdk::Error::loading("tokens must have rank one"))?;
        if sequence == 0 {
            return Err(forja_sdk::Error::loading("tokens cannot be empty"));
        }
        let end = input
            .start_pos
            .checked_add(sequence)
            .filter(|&end| end <= MAX_CONTEXT)
            .ok_or_else(|| forja_sdk::Error::loading("tokens exceed the 4096-token context"))?;
        #[cfg(target_family = "wasm")]
        if REPLAY_DECODE && sequence == 1 && !input.taps {
            let logits =
                self.replay_decode(Some(&input.tokens), input.start_pos, DecodeSelection::None)?;
            return Ok(StepOutput {
                logits,
                taps: Vec::new(),
                router_logits: Vec::new(),
            });
        }
        #[cfg(target_family = "wasm")]
        if sequence > 1 && !input.taps && self.prefill_supports(input.start_pos, sequence) {
            let logits = if REPLAY_DECODE {
                self.replay_prefill(&input.tokens, input.start_pos)?
            } else {
                self.lazy_prefill(&input.tokens, input.start_pos)?
            };
            return Ok(StepOutput {
                logits,
                taps: Vec::new(),
                router_logits: Vec::new(),
            });
        }
        let start = input.start_pos.into();
        let end = end.into();
        let attention = attention_variant(sequence, input.start_pos, None)?;
        self.forward(
            &input.tokens,
            sequence,
            &start,
            &end,
            input.taps,
            &attention,
        )
    }

    #[cfg(target_family = "wasm")]
    fn decode(&mut self, input: forja_sdk::DecodeInput) -> Result<forja_sdk::DecodeOutput> {
        let selection = self.decode.select(input.sampling)?;
        match input.tokens {
            Some(tokens) => {
                let [sequence] = tokens
                    .shape()
                    .try_into()
                    .map_err(|_| forja_sdk::Error::loading("decode tokens must have rank one"))?;
                if sequence == 0 {
                    return Err(forja_sdk::Error::loading("decode tokens cannot be empty"));
                }
                let end = input
                    .start_pos
                    .checked_add(sequence)
                    .filter(|&end| end <= MAX_CONTEXT)
                    .ok_or_else(|| {
                        forja_sdk::Error::loading("decode tokens exceed the 4096-token context")
                    })?;
                if self.prefill_supports(input.start_pos, sequence) {
                    let logits = if REPLAY_DECODE {
                        self.replay_prefill(&tokens, input.start_pos)?
                    } else {
                        self.lazy_prefill(&tokens, input.start_pos)?
                    };
                    self.select_token(logits, end - 1)
                } else if REPLAY_DECODE && sequence == 1 {
                    self.decode_selected(Some(&tokens), input.start_pos, selection)
                } else {
                    let start = input.start_pos.into();
                    let end_dim = end.into();
                    let attention = attention_variant(sequence, input.start_pos, None)?;
                    let logits = self
                        .forward(&tokens, sequence, &start, &end_dim, false, &attention)?
                        .logits;
                    self.select_token(logits, end - 1)
                }
            }
            None if input.start_pos < MAX_CONTEXT && REPLAY_DECODE => {
                self.decode_selected(None, input.start_pos, selection)
            }
            None if input.start_pos < MAX_CONTEXT => {
                let token = self.decode.token()?;
                let start = input.start_pos.into();
                let end = (input.start_pos + 1).into();
                let attention = attention_variant(1, input.start_pos, None)?;
                let logits = self
                    .forward(&token, 1, &start, &end, false, &attention)?
                    .logits;
                self.select_token(logits, input.start_pos)
            }
            None => Err(forja_sdk::Error::loading(
                "decode position exceeds the 4096-token context",
            )),
        }
    }
}

#[cfg(target_family = "wasm")]
impl Olmoe {
    fn prefill_supports(&self, start: u32, sequence: u32) -> bool {
        self.prefill
            .as_ref()
            .is_some_and(|prefill| prefill.supports(start, sequence))
    }

    fn replay_prefill(&mut self, tokens: &Tensor<u32>, start: u32) -> Result<Tensor<f32>> {
        let mut prefill = self
            .prefill
            .take()
            .ok_or_else(|| forja_sdk::Error::loading("prefill state is unavailable"))?;
        let result = prefill.replay(
            tokens,
            start,
            |tokens, sequence, last, start, end, parameter| {
                self.forward_prefill_chunk(
                    tokens,
                    sequence,
                    last,
                    start,
                    end,
                    PrefillVariant {
                        start: 0,
                        parameter: Some(parameter),
                    },
                )
            },
        );
        self.prefill = Some(prefill);
        result
    }

    fn lazy_prefill(&mut self, tokens: &Tensor<u32>, start: u32) -> Result<Tensor<f32>> {
        let prefill = self
            .prefill
            .take()
            .ok_or_else(|| forja_sdk::Error::loading("prefill state is unavailable"))?;
        let result = prefill.lazy(
            tokens,
            start,
            |tokens, sequence, last, start, end, start_value| {
                self.forward_prefill_chunk(
                    tokens,
                    sequence,
                    last,
                    start,
                    end,
                    PrefillVariant {
                        start: start_value,
                        parameter: None,
                    },
                )
            },
        );
        self.prefill = Some(prefill);
        result
    }

    fn replay_decode(
        &mut self,
        tokens: Option<&Tensor<u32>>,
        start_pos: u32,
        selection: DecodeSelection,
    ) -> Result<Tensor<f32>> {
        if let Some(tokens) = tokens {
            self.decode.write_token(tokens)?;
        }
        if self.decode.graph(selection).is_none() {
            let graph = self.capture_decode(start_pos, selection)?;
            self.decode.set_graph(selection, graph);
        }
        let graph = self
            .decode
            .graph(selection)
            .ok_or_else(|| forja_sdk::Error::loading("decode graph was not captured"))?;
        graph.replay(&[start_pos])?;
        graph.result().alias()
    }

    fn capture_decode(
        &mut self,
        start_pos: u32,
        selection: DecodeSelection,
    ) -> Result<forja_sdk::Graph<Tensor<f32>>> {
        let position = forja_sdk::Param::new(0..=MAX_CONTEXT - 1)?;
        let start = position.at(start_pos);
        let end = (start.clone() + 1)?;
        let start_dim = start.clone().into();
        let token = self.decode.token()?;
        let mut feedback = self.decode.token()?;
        let output_tokens = self.decode.output_tokens()?;
        let sampling = self.decode.sampling()?;
        forja_sdk::capture(&[&position], || {
            let attention = attention_variant(1, start_pos, Some(&position))?;
            let logits = self
                .forward(&token, 1, &start_dim, &end, false, &attention)?
                .logits;
            let selected = match selection {
                DecodeSelection::None => None,
                DecodeSelection::Greedy => Some(logits.reshape(&[1, VOCAB])?.argmax()?),
                DecodeSelection::Sampled => Some(
                    logits
                        .reshape(&[1, VOCAB])?
                        .sample(&sampling, start.clone())?,
                ),
            };
            if let Some(selected) = selected {
                selected.copy_into(&mut feedback)?;
                selected.copy_into(&mut output_tokens.narrow(0, start, 1)?)?;
            }
            Ok(logits)
        })
    }

    fn decode_selected(
        &mut self,
        tokens: Option<&Tensor<u32>>,
        start_pos: u32,
        selection: DecodeSelection,
    ) -> Result<forja_sdk::DecodeOutput> {
        let logits = self.replay_decode(tokens, start_pos, selection)?;
        Ok(forja_sdk::DecodeOutput {
            logits,
            token: self.decode.selected(start_pos)?,
        })
    }

    fn select_token(&mut self, logits: Tensor<f32>, slot: u32) -> Result<forja_sdk::DecodeOutput> {
        let selected = logits
            .reshape(&[1, VOCAB])?
            .sample(&self.decode.sampling()?, slot)?;
        let token = self.decode.store_selected(&selected, slot)?;
        Ok(forja_sdk::DecodeOutput { logits, token })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn describes_model_contract() {
        let info = Olmoe::describe();
        assert_eq!(info.vocab, VOCAB);
        assert_eq!(info.max_context, MAX_CONTEXT);
        assert_eq!(info.tap_layers, (1..=16).collect::<Vec<_>>());
        assert_eq!(info.router_layers, (1..=16).collect::<Vec<_>>());
    }

    #[test]
    fn selects_dense_variants_at_algorithm_boundaries() {
        assert_eq!(
            matmul_variant(DType::F32, 1, 1, HIDDEN, HIDDEN, true)
                .unwrap()
                .name(),
            "matmul.gemv-transposed"
        );
        assert_eq!(
            matmul_variant(DType::F32, 1, 511, HIDDEN, HIDDEN, true)
                .unwrap()
                .name(),
            "matmul.steel-32x64x16-1x2"
        );
        assert_eq!(
            matmul_variant(DType::F32, 1, 512, HIDDEN, HIDDEN, true)
                .unwrap()
                .name(),
            "matmul.steel-64x64x16-2x2"
        );
        assert_eq!(
            matmul_variant(DType::BF16, 1, 512, VOCAB, HIDDEN, true)
                .unwrap()
                .name(),
            "matmul.steel-64x64x16-1x2"
        );
    }

    #[test]
    fn selects_attention_variants_at_cache_boundaries() {
        let names = [510, 511, 1022, 1023].map(|start| match attention_variant(1, start, None) {
            Ok(AttentionVariant::Fixed(variant)) => variant.name().to_owned(),
            Ok(AttentionVariant::Rule(_)) => panic!("expected a concrete variant"),
            Err(error) => panic!("variant selection failed: {error}"),
        });
        assert_eq!(
            names,
            [
                "sdpa.decomposed",
                "sdpa.vector-single-pass",
                "sdpa.vector-single-pass",
                "sdpa.vector-two-pass",
            ]
        );
    }
}
