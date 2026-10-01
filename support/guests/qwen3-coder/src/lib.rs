//! Qwen3-Coder-30B-A3B-Instruct MLX 4-bit inference engine.

#[cfg(target_family = "wasm")]
use forja_sdk::nn::blocks::{ChunkedPrefill, DEFAULT_PREFILL_CHUNK, DecodeSelection, DecodeState};
use forja_sdk::{
    DType, Dim, Engine, EngineInfo, EngineLoadConfig, Load, Result, StepInput, StepOutput, Tensor,
    Weights, bf16, export_engine,
    kernel::Kernel,
    nn::{
        RmsNorm, RmsNormConfig,
        blocks::{KvCache, Taps, cached_attention, residual_norm, residual_norm_kernel},
        moe_combine, moe_router,
    },
};

/// Vocabulary size of Qwen3-Coder-30B-A3B-Instruct.
pub const VOCAB: u32 = 151_936;
/// Hidden width of Qwen3-Coder-30B-A3B-Instruct.
pub const HIDDEN: u32 = 2_048;
/// Number of decoder layers in the full model.
pub const LAYERS: usize = 48;
/// Largest context accepted by this engine.
pub const MAX_CONTEXT: u32 = 4_096;

const QUERY_HEADS: u32 = 32;
const KEY_VALUE_HEADS: u32 = 4;
const HEAD_DIM: u32 = 128;
const INTERMEDIATE: u32 = 768;
/// Number of routed experts.
pub const EXPERTS: u32 = 128;
/// Experts selected for each token.
pub const TOP_K: u32 = 8;
const RMS_EPSILON: f32 = 1.0e-6;
const ROPE_THETA: f32 = 10_000_000.0;
const ATTENTION_SCALE: f32 = 0.088_388_35;
const Q4_BITS: u8 = 4;
const Q8_BITS: u8 = 8;
const QUANT_GROUP: u32 = 64;
#[cfg(target_family = "wasm")]
const REPLAY_DECODE: bool = !cfg!(feature = "no-replay");

#[derive(Clone, Copy)]
struct Config;

type SequenceOutput = (Tensor<f32>, Vec<Tensor<f32>>, Vec<Tensor<f32>>);

#[derive(Clone, Copy)]
struct QuantConfig {
    input: u32,
    output: u32,
    bits: u8,
}

struct QuantLinear {
    packed: Tensor<u32>,
    scales: Tensor<bf16>,
    biases: Tensor<bf16>,
    bits: u8,
}

impl Load<QuantConfig> for QuantLinear {
    fn load(weights: &Weights<'_>, config: &QuantConfig) -> Result<Self> {
        let packed_width = config
            .input
            .checked_mul(u32::from(config.bits))
            .and_then(|width| width.checked_div(32))
            .ok_or_else(|| forja_sdk::Error::loading("quantized width overflowed"))?;
        Ok(Self {
            packed: weights.tensor("weight", &[config.output, packed_width])?,
            scales: weights.tensor("scales", &[config.output, config.input / QUANT_GROUP])?,
            biases: weights.tensor("biases", &[config.output, config.input / QUANT_GROUP])?,
            bits: config.bits,
        })
    }
}

impl QuantLinear {
    fn forward(&self, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        input.quant_matmul(
            &self.packed,
            &self.scales,
            &self.biases,
            self.bits,
            QUANT_GROUP,
        )
    }
}

struct QuantEmbedding(QuantLinear);

impl QuantEmbedding {
    fn load(weights: &Weights<'_>) -> Result<Self> {
        Ok(Self(QuantLinear::load(
            weights,
            &QuantConfig {
                input: HIDDEN,
                output: VOCAB,
                bits: Q4_BITS,
            },
        )?))
    }

    fn forward(&self, ids: &Tensor<u32>) -> Result<Tensor<f32>> {
        ids.quant_embedding(
            &self.0.packed,
            &self.0.scales,
            &self.0.biases,
            self.0.bits,
            QUANT_GROUP,
        )
    }
}

#[derive(Load)]
#[load(config = Config)]
struct Attention {
    #[load(prefix, config = QuantConfig { input: HIDDEN, output: QUERY_HEADS * HEAD_DIM, bits: Q4_BITS })]
    q_proj: QuantLinear,
    #[load(prefix, config = QuantConfig { input: HIDDEN, output: KEY_VALUE_HEADS * HEAD_DIM, bits: Q4_BITS })]
    k_proj: QuantLinear,
    #[load(prefix, config = QuantConfig { input: HIDDEN, output: KEY_VALUE_HEADS * HEAD_DIM, bits: Q4_BITS })]
    v_proj: QuantLinear,
    #[load(prefix, config = QuantConfig { input: QUERY_HEADS * HEAD_DIM, output: HIDDEN, bits: Q4_BITS })]
    o_proj: QuantLinear,
    #[load(prefix, config = RmsNormConfig::promoted_bf16(HEAD_DIM, RMS_EPSILON))]
    q_norm: RmsNorm<f32>,
    #[load(prefix, config = RmsNormConfig::promoted_bf16(HEAD_DIM, RMS_EPSILON))]
    k_norm: RmsNorm<f32>,
}

struct QuantExperts {
    packed: Tensor<u32>,
    scales: Tensor<bf16>,
    biases: Tensor<bf16>,
}

impl QuantExperts {
    fn load(weights: &Weights<'_>, input: u32, output: u32) -> Result<Self> {
        Ok(Self {
            packed: weights.tensor("weight", &[EXPERTS, output, input / 8])?,
            scales: weights.tensor("scales", &[EXPERTS, output, input / QUANT_GROUP])?,
            biases: weights.tensor("biases", &[EXPERTS, output, input / QUANT_GROUP])?,
        })
    }

    fn forward(&self, input: &Tensor<f32>, indices: &Tensor<u32>) -> Result<Tensor<f32>> {
        input.gather_quant_matmul(
            &self.packed,
            &self.scales,
            &self.biases,
            indices,
            Q4_BITS,
            QUANT_GROUP,
        )
    }
}

struct SparseMoe {
    gate: QuantLinear,
    gate_proj: QuantExperts,
    up_proj: QuantExperts,
    down_proj: QuantExperts,
}

impl Load<Config> for SparseMoe {
    fn load(weights: &Weights<'_>, _config: &Config) -> Result<Self> {
        let switch = weights.scoped("switch_mlp");
        Ok(Self {
            gate: QuantLinear::load(
                &weights.scoped("gate"),
                &QuantConfig {
                    input: HIDDEN,
                    output: EXPERTS,
                    bits: Q8_BITS,
                },
            )?,
            gate_proj: QuantExperts::load(&switch.scoped("gate_proj"), HIDDEN, INTERMEDIATE)?,
            up_proj: QuantExperts::load(&switch.scoped("up_proj"), HIDDEN, INTERMEDIATE)?,
            down_proj: QuantExperts::load(&switch.scoped("down_proj"), INTERMEDIATE, HIDDEN)?,
        })
    }
}

impl SparseMoe {
    fn forward(&self, input: &Tensor<f32>) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let logits = self.gate.forward(input)?;
        let (route_weights, indices) = moe_router(&logits, TOP_K, true)?;
        let activated = input.gather_quant_silu_mul(
            &self.gate_proj.packed,
            &self.gate_proj.scales,
            &self.gate_proj.biases,
            &self.up_proj.packed,
            &self.up_proj.scales,
            &self.up_proj.biases,
            &indices,
            Q4_BITS,
            QUANT_GROUP,
        )?;
        let rows = input
            .shape()
            .first()
            .copied()
            .and_then(|rows| rows.checked_mul(TOP_K))
            .ok_or_else(|| forja_sdk::Error::loading("routed row count overflowed"))?;
        let output = self
            .down_proj
            .forward(
                &activated.reshape(&[rows, INTERMEDIATE])?,
                &indices.reshape(&[rows, 1])?,
            )?
            .reshape(&[rows / TOP_K, TOP_K, HIDDEN])?;
        Ok((moe_combine(&output, &route_weights)?, logits))
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

struct LayerResources<'a> {
    cache: &'a mut KvCache<f32>,
    following_norm: &'a RmsNorm<f32>,
}

#[derive(Clone, Copy)]
struct LayerPosition<'a> {
    positions: &'a Tensor<u32>,
    start: &'a Dim,
    end: &'a Dim,
}

impl DecoderLayer {
    fn forward(
        &self,
        residual_kernel: &Kernel,
        input: &Tensor<f32>,
        normalized_input: &Tensor<f32>,
        position: LayerPosition<'_>,
        resources: LayerResources<'_>,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
        let LayerResources {
            cache,
            following_norm,
        } = resources;
        let LayerPosition {
            positions,
            start,
            end,
        } = position;
        let sequence = positions.shape()[0];
        let query = self
            .self_attn
            .q_norm
            .forward(&self.self_attn.q_proj.forward(normalized_input)?.reshape(&[
                sequence,
                QUERY_HEADS,
                HEAD_DIM,
            ])?)?
            .rope(positions, ROPE_THETA)?;
        let key = self
            .self_attn
            .k_norm
            .forward(&self.self_attn.k_proj.forward(normalized_input)?.reshape(&[
                sequence,
                KEY_VALUE_HEADS,
                HEAD_DIM,
            ])?)?
            .rope(positions, ROPE_THETA)?;
        let value = self.self_attn.v_proj.forward(normalized_input)?.reshape(&[
            sequence,
            KEY_VALUE_HEADS,
            HEAD_DIM,
        ])?;
        let attended = cached_attention(&query, &key, &value, cache, ATTENTION_SCALE, start, end)?;
        let (hidden, normalized) = residual_norm(
            residual_kernel,
            input,
            &self.self_attn.o_proj.forward(&attended)?,
            self.post_attention_layernorm.weight(),
        )?;
        let (projected, router_logits) = self.mlp.forward(&normalized)?;
        let (hidden, normalized) = residual_norm(
            residual_kernel,
            &hidden,
            &projected,
            following_norm.weight(),
        )?;
        Ok((hidden, normalized, router_logits))
    }
}

struct QwenWeights {
    embed_tokens: QuantEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm<f32>,
    lm_head: QuantLinear,
}

impl QwenWeights {
    fn load(weights: &Weights<'_>, layers: usize) -> Result<Self> {
        let model = weights.scoped("model");
        let layer_weights = model.scoped("layers");
        Ok(Self {
            embed_tokens: QuantEmbedding::load(&model.scoped("embed_tokens"))?,
            layers: (0..layers)
                .map(|index| DecoderLayer::load(&layer_weights.scoped(index.to_string()), &Config))
                .collect::<Result<Vec<_>>>()?,
            norm: RmsNorm::load(
                &model.scoped("norm"),
                &RmsNormConfig::promoted_bf16(HIDDEN, RMS_EPSILON),
            )?,
            lm_head: QuantLinear::load(
                &weights.scoped("lm_head"),
                &QuantConfig {
                    input: HIDDEN,
                    output: VOCAB,
                    bits: Q4_BITS,
                },
            )?,
        })
    }
}

/// Qwen3-Coder-30B-A3B-Instruct with a fixed 4096-token KV cache.
pub struct Qwen3Coder {
    weights: QwenWeights,
    caches: Vec<KvCache<f32>>,
    residual_norm: Kernel,
    positions: Tensor<u32>,
    #[cfg(target_family = "wasm")]
    decode: DecodeState,
    #[cfg(target_family = "wasm")]
    prefill: Option<ChunkedPrefill<f32>>,
}

impl Qwen3Coder {
    fn load_from_weights(weights: &Weights<'_>, layers: usize) -> Result<Self> {
        let weights = QwenWeights::load(weights, layers)?;
        let caches = (0..layers)
            .map(|_| KvCache::new(KEY_VALUE_HEADS, MAX_CONTEXT, HEAD_DIM, 0.0))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights,
            caches,
            residual_norm: residual_norm_kernel(DType::F32, RMS_EPSILON)?,
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
    ) -> Result<StepOutput> {
        let (logits, taps, router_logits) =
            self.forward_sequence(tokens, sequence, start, end, taps_enabled)?;
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
    ) -> Result<SequenceOutput> {
        let positions = self.positions.narrow(0, start, sequence)?;
        let mut hidden = self.weights.embed_tokens.forward(tokens)?;
        let layers = self.weights.layers.len();
        let mut normalized = self.weights.layers[0].input_layernorm.forward(&hidden)?;
        let mut taps = Taps::new(taps_enabled, layers);
        let mut routers = Vec::with_capacity(if taps_enabled { layers } else { 0 });
        for index in 0..layers {
            let following_norm = self
                .weights
                .layers
                .get(index + 1)
                .map_or(&self.weights.norm, |layer| &layer.input_layernorm);
            let (next, next_normalized, router) = self.weights.layers[index].forward(
                &self.residual_norm,
                &hidden,
                &normalized,
                LayerPosition {
                    positions: &positions,
                    start,
                    end,
                },
                LayerResources {
                    cache: &mut self.caches[index],
                    following_norm,
                },
            )?;
            hidden = next;
            normalized = next_normalized;
            if taps.enabled() {
                routers.push(router.contiguous()?);
                if index + 1 < layers {
                    taps.push(hidden.contiguous()?);
                }
            }
        }
        if taps.enabled() {
            taps.push(normalized.contiguous()?);
        }
        let logits = self.weights.lm_head.forward(&normalized)?;
        Ok((logits, taps.finish(), routers))
    }

    fn last_logits(logits: &Tensor<f32>, last: impl Into<Dim>) -> Result<Tensor<f32>> {
        logits.narrow(0, last, 1)?.reshape(&[VOCAB])?.contiguous()
    }

    #[cfg(target_family = "wasm")]
    fn forward_prefill_chunk(
        &mut self,
        tokens: &Tensor<u32>,
        sequence: u32,
        last: &Dim,
        start: &Dim,
        end: &Dim,
    ) -> Result<Tensor<f32>> {
        let (logits, _, _) = self.forward_sequence(tokens, sequence, start, end, false)?;
        Self::last_logits(&logits, last)
    }
}

#[export_engine]
impl Engine for Qwen3Coder {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: VOCAB,
            max_context: MAX_CONTEXT,
            tap_layers: (1..=48).collect(),
            router_layers: (1..=48).collect(),
        }
    }

    fn load(weights: &Weights<'_>, config: EngineLoadConfig) -> Result<Self> {
        let layers = config
            .num_hidden_layers
            .map_or(Ok(LAYERS), usize::try_from)
            .map_err(|_| forja_sdk::Error::loading("layer count does not fit usize"))?;
        if !(1..=LAYERS).contains(&layers) {
            return Err(forja_sdk::Error::loading(
                "layer count must be between 1 and 48",
            ));
        }
        Self::load_from_weights(weights, layers)
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
        self.forward(&input.tokens, sequence, &start, &end, input.taps)
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
                    let logits = self
                        .forward(&tokens, sequence, &start, &end_dim, false)?
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
                let logits = self.forward(&token, 1, &start, &end, false)?.logits;
                self.select_token(logits, input.start_pos)
            }
            None => Err(forja_sdk::Error::loading(
                "decode position exceeds the 4096-token context",
            )),
        }
    }
}

#[cfg(target_family = "wasm")]
impl Qwen3Coder {
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
        let result = prefill.replay(tokens, start, |tokens, sequence, last, start, end| {
            self.forward_prefill_chunk(tokens, sequence, last, start, end)
        });
        self.prefill = Some(prefill);
        result
    }

    fn lazy_prefill(&mut self, tokens: &Tensor<u32>, start: u32) -> Result<Tensor<f32>> {
        let prefill = self
            .prefill
            .take()
            .ok_or_else(|| forja_sdk::Error::loading("prefill state is unavailable"))?;
        let result = prefill.lazy(tokens, start, |tokens, sequence, last, start, end| {
            self.forward_prefill_chunk(tokens, sequence, last, start, end)
        });
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
            let logits = self.forward(&token, 1, &start_dim, &end, false)?.logits;
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
        let info = Qwen3Coder::describe();
        assert_eq!(info.vocab, VOCAB);
        assert_eq!(info.max_context, MAX_CONTEXT);
        assert_eq!(info.tap_layers, (1..=48).collect::<Vec<_>>());
        assert_eq!(info.router_layers, (1..=48).collect::<Vec<_>>());
    }
}
