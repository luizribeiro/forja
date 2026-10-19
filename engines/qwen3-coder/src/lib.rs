//! Qwen3-Coder-30B-A3B-Instruct MLX 4-bit inference engine.

mod variants;

#[cfg(target_family = "wasm")]
use forja_sdk::Param;
#[cfg(target_family = "wasm")]
use forja_sdk::nn::blocks::{ChunkedPrefill, DecodeSelection, DecodeState, PrefillExecution};
use forja_sdk::{
    CachePlane, DType, Dim, Engine, EngineInfo, EngineLoadConfig, KvCacheFormat, Load, Result,
    ServingInfo, StepInput, StepOutput, Tensor, VariantSelection, Weights, bf16, export_engine,
    f16,
    kernel::{Epilogue, Kernel, TensorRef},
    nn::{
        RmsNorm, RmsNormConfig,
        blocks::{
            KvCache, Taps, cached_attention_with, qk_norm_rope, qk_norm_rope_kernel, residual_norm,
            residual_norm_kernel, rms_norm, rms_norm_kernel,
        },
        moe_combine_with, moe_router_with,
    },
};

#[forja_sdk::kernel(map)]
fn moe_gate_up(
    gate: forja_sdk::kernel::Elem,
    up: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    gate * gate.sigmoid() * up
}

#[forja_sdk::kernel(map)]
fn moe_add_residual(
    accumulator: forja_sdk::kernel::Elem,
    residual: forja_sdk::kernel::Elem,
) -> forja_sdk::kernel::Elem {
    accumulator + residual
}

/// Vocabulary size of Qwen3-Coder-30B-A3B-Instruct.
pub const VOCAB: u32 = 151_936;
/// Hidden width of Qwen3-Coder-30B-A3B-Instruct.
pub const HIDDEN: u32 = 2_048;
/// Number of decoder layers in the full model.
pub const LAYERS: usize = 48;
/// Largest context accepted by this engine.
pub const MAX_CONTEXT: u32 = 32_768;
#[cfg(target_family = "wasm")]
const PREFILL_CHUNK: u32 = 2_048;
#[cfg(target_family = "wasm")]
const PREFILL_TAIL_CHUNK: u32 = 16;
#[cfg(target_family = "wasm")]
const SHORT_PREFILL_CHUNK: u32 = 512;
#[cfg(target_family = "wasm")]
const SHORT_PREFILL_TAIL_CAPACITY: usize = 5;
#[cfg(target_family = "wasm")]
const LONG_PREFILL_TAIL_CAPACITY: usize = 2;

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

fn serving_info() -> ServingInfo {
    let planes = (0..LAYERS)
        .flat_map(|layer| {
            ["key", "value"].map(move |kind| CachePlane {
                elements_per_token: u64::from(KEY_VALUE_HEADS * HEAD_DIM),
                alignment: 256,
                label: format!("layer-{layer}-{kind}"),
            })
        })
        .collect();
    ServingInfo {
        page_tokens: 64,
        planes,
        active_lanes: 8,
        max_context: MAX_CONTEXT,
    }
}

fn exact_position(position: u32) -> Result<f32> {
    if position > 1 << f32::MANTISSA_DIGITS {
        return Err(forja_sdk::Error::loading(
            "position exceeds the exact f32 integer range",
        ));
    }
    #[allow(clippy::cast_precision_loss)]
    Ok(position as f32)
}

#[derive(Clone, Copy)]
struct Config;

type SequenceOutput = (Tensor<f32>, Vec<Tensor<f32>>, Vec<Tensor<f32>>);

#[cfg(target_family = "wasm")]
#[derive(Clone, Copy)]
struct PrefillVariant<'a> {
    start: u32,
    parameter: Option<&'a Param>,
}

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
    fn forward(&self, variants: &variants::Variants, input: &Tensor<f32>) -> Result<Tensor<f32>> {
        let rows = input.shape().first().copied().ok_or_else(|| {
            forja_sdk::Error::loading("quantized linear input must have rank two")
        })?;
        let selection = variants.quant(rows, self.bits)?;
        input.quant_matmul_with(
            &self.packed,
            &self.scales,
            &self.biases,
            self.bits,
            QUANT_GROUP,
            &selection,
        )
    }

    fn load_stacked(
        weights: &Weights<'_>,
        projections: &[(&str, u32)],
        input: u32,
        bits: u8,
    ) -> Result<Self> {
        let packed_width = input
            .checked_mul(u32::from(bits))
            .and_then(|width| width.checked_div(32))
            .ok_or_else(|| forja_sdk::Error::loading("quantized width overflowed"))?;
        Ok(Self {
            packed: stack_projection(weights, projections, "weight", packed_width)?,
            scales: stack_projection(weights, projections, "scales", input / QUANT_GROUP)?,
            biases: stack_projection(weights, projections, "biases", input / QUANT_GROUP)?,
            bits,
        })
    }

    fn output_slice(&self, offset: u32, output: u32) -> Result<Self> {
        Ok(Self {
            packed: self.packed.narrow(0, offset, output)?,
            scales: self.scales.narrow(0, offset, output)?,
            biases: self.biases.narrow(0, offset, output)?,
            bits: self.bits,
        })
    }
}

fn stack_projection<T: forja_sdk::Element>(
    weights: &Weights<'_>,
    projections: &[(&str, u32)],
    name: &str,
    width: u32,
) -> Result<Tensor<T>> {
    let rows = projections.iter().try_fold(0_u32, |total, (_, rows)| {
        total
            .checked_add(*rows)
            .ok_or_else(|| forja_sdk::Error::loading("stacked projection height overflowed"))
    })?;
    let output = Tensor::zeros(&[rows, width])?;
    let mut offset = 0_u32;
    for (prefix, rows) in projections {
        let source = weights.scoped(prefix).tensor::<T>(name, &[*rows, width])?;
        let mut destination = output.narrow(0, offset, *rows)?;
        source.copy_into(&mut destination)?;
        offset = offset
            .checked_add(*rows)
            .ok_or_else(|| forja_sdk::Error::loading("stacked projection offset overflowed"))?;
    }
    Ok(output)
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

struct Attention {
    qkv_proj: QuantLinear,
    q_proj: QuantLinear,
    k_proj: QuantLinear,
    v_proj: QuantLinear,
    o_proj: QuantLinear,
    q_norm: RmsNorm<f32>,
    k_norm: RmsNorm<f32>,
    decode_norm: Tensor<f32>,
}

impl Load<Config> for Attention {
    fn load(weights: &Weights<'_>, _config: &Config) -> Result<Self> {
        let qkv_proj = QuantLinear::load_stacked(
            weights,
            &[
                ("q_proj", QUERY_HEADS * HEAD_DIM),
                ("k_proj", KEY_VALUE_HEADS * HEAD_DIM),
                ("v_proj", KEY_VALUE_HEADS * HEAD_DIM),
            ],
            HIDDEN,
            Q4_BITS,
        )?;
        let q_norm = RmsNorm::load(
            &weights.scoped("q_norm"),
            &RmsNormConfig::promoted_bf16(HEAD_DIM, RMS_EPSILON),
        )?;
        let k_norm = RmsNorm::load(
            &weights.scoped("k_norm"),
            &RmsNormConfig::promoted_bf16(HEAD_DIM, RMS_EPSILON),
        )?;
        let decode_norm = stack_decode_norm(q_norm.weight(), k_norm.weight())?;
        Ok(Self {
            q_proj: qkv_proj.output_slice(0, QUERY_HEADS * HEAD_DIM)?,
            k_proj: qkv_proj.output_slice(QUERY_HEADS * HEAD_DIM, KEY_VALUE_HEADS * HEAD_DIM)?,
            v_proj: qkv_proj.output_slice(
                (QUERY_HEADS + KEY_VALUE_HEADS) * HEAD_DIM,
                KEY_VALUE_HEADS * HEAD_DIM,
            )?,
            qkv_proj,
            o_proj: QuantLinear::load(
                &weights.scoped("o_proj"),
                &QuantConfig {
                    input: QUERY_HEADS * HEAD_DIM,
                    output: HIDDEN,
                    bits: Q4_BITS,
                },
            )?,
            q_norm,
            k_norm,
            decode_norm,
        })
    }
}

impl Attention {
    fn project_normalized(
        &self,
        variants: &variants::Variants,
        input: &Tensor<f32>,
        sequence: u32,
        kernel: &Kernel,
        positions: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
        if sequence == 1 {
            let qkv = self.qkv_proj.forward(variants, input)?;
            let qk = qk_norm_rope(
                kernel,
                &qkv.narrow(1, 0, (QUERY_HEADS + KEY_VALUE_HEADS) * HEAD_DIM)?
                    .reshape(&[sequence, QUERY_HEADS + KEY_VALUE_HEADS, HEAD_DIM])?,
                &self.decode_norm,
                positions,
            )?;
            return Ok((
                qk.narrow(1, 0, QUERY_HEADS)?,
                qk.narrow(1, QUERY_HEADS, KEY_VALUE_HEADS)?,
                qkv.narrow(
                    1,
                    (QUERY_HEADS + KEY_VALUE_HEADS) * HEAD_DIM,
                    KEY_VALUE_HEADS * HEAD_DIM,
                )?
                .reshape(&[sequence, KEY_VALUE_HEADS, HEAD_DIM])?,
            ));
        }
        let query = self.q_proj.forward(variants, input)?;
        let key = self.k_proj.forward(variants, input)?;
        Ok((
            qk_norm_rope(
                kernel,
                &query.reshape(&[sequence, QUERY_HEADS, HEAD_DIM])?,
                self.q_norm.weight(),
                positions,
            )?,
            qk_norm_rope(
                kernel,
                &key.reshape(&[sequence, KEY_VALUE_HEADS, HEAD_DIM])?,
                self.k_norm.weight(),
                positions,
            )?,
            self.v_proj.forward(variants, input)?.reshape(&[
                sequence,
                KEY_VALUE_HEADS,
                HEAD_DIM,
            ])?,
        ))
    }
}

fn stack_decode_norm(query: &Tensor<f32>, key: &Tensor<f32>) -> Result<Tensor<f32>> {
    let output = Tensor::zeros(&[QUERY_HEADS + KEY_VALUE_HEADS, HEAD_DIM])?;
    let mut query_output = output.narrow(0, 0, QUERY_HEADS)?;
    query
        .broadcast_as(&[QUERY_HEADS, HEAD_DIM])?
        .copy_into(&mut query_output)?;
    let mut key_output = output.narrow(0, QUERY_HEADS, KEY_VALUE_HEADS)?;
    key.broadcast_as(&[KEY_VALUE_HEADS, HEAD_DIM])?
        .copy_into(&mut key_output)?;
    Ok(output)
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

    fn forward(
        &self,
        variants: &variants::Variants,
        input: &Tensor<f32>,
        indices: &Tensor<u32>,
    ) -> Result<Tensor<f32>> {
        let [rows, routes]: [u32; 2] = indices.shape().try_into().map_err(|_| {
            forja_sdk::Error::loading("gathered quantized indices must have rank two")
        })?;
        let routed_rows = rows
            .checked_mul(routes)
            .ok_or_else(|| forja_sdk::Error::loading("routed row count overflowed"))?;
        let selection = variants.gather_matmul(routed_rows)?;
        input.gather_quant_matmul_with(
            &self.packed,
            &self.scales,
            &self.biases,
            indices,
            Q4_BITS,
            QUANT_GROUP,
            &selection,
        )
    }
}

struct SparseMoe {
    gate: QuantLinear,
    gate_proj: QuantExperts,
    up_proj: QuantExperts,
    down_proj: QuantExperts,
}

struct MoeEpilogues {
    gate_up: Epilogue,
    add_residual: Epilogue,
}

impl MoeEpilogues {
    fn load() -> Result<Self> {
        Ok(Self {
            gate_up: moe_gate_up_epilogue(2, 3, &[], DType::F32)?,
            add_residual: moe_add_residual_epilogue(1, 2, &[DType::F32], DType::F32)?,
        })
    }
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
    fn route(
        &self,
        variants: &variants::Variants,
        epilogues: &MoeEpilogues,
        input: &Tensor<f32>,
    ) -> Result<RoutedMoe> {
        let logits = self.gate.forward(variants, input)?;
        let top_k = variants.top_k()?;
        let (weights, indices) = moe_router_with(&logits, TOP_K, true, &top_k)?;
        let rows = input
            .shape()
            .first()
            .copied()
            .ok_or_else(|| forja_sdk::Error::loading("MoE input must have rank two"))?;
        let routed_rows = rows
            .checked_mul(TOP_K)
            .ok_or_else(|| forja_sdk::Error::loading("routed row count overflowed"))?;
        let selection = variants.gate_up(routed_rows)?;
        let activated = input.gather_quant_matmul_pair_epilogue_with(
            [
                (
                    &self.gate_proj.packed,
                    &self.gate_proj.scales,
                    &self.gate_proj.biases,
                ),
                (
                    &self.up_proj.packed,
                    &self.up_proj.scales,
                    &self.up_proj.biases,
                ),
            ],
            &indices,
            Q4_BITS,
            QUANT_GROUP,
            &epilogues.gate_up,
            &[],
            &selection,
        )?;
        Ok(RoutedMoe {
            logits,
            weights,
            indices,
            activated,
        })
    }

    fn forward(
        &self,
        variants: &variants::Variants,
        epilogues: &MoeEpilogues,
        input: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let routed = self.route(variants, epilogues, input)?;
        let rows = input
            .shape()
            .first()
            .copied()
            .and_then(|rows| rows.checked_mul(TOP_K))
            .ok_or_else(|| forja_sdk::Error::loading("routed row count overflowed"))?;
        let output = self
            .down_proj
            .forward(
                variants,
                &routed.activated.reshape(&[rows, INTERMEDIATE])?,
                &routed.indices.reshape(&[rows, 1])?,
            )?
            .reshape(&[rows / TOP_K, TOP_K, HIDDEN])?;
        let selection = variants.combine(rows / TOP_K)?;
        Ok((
            moe_combine_with(&output, &routed.weights, &selection)?,
            routed.logits,
        ))
    }

    fn forward_decode(
        &self,
        variants: &variants::Variants,
        epilogues: &MoeEpilogues,
        input: &Tensor<f32>,
        residual: &Tensor<f32>,
    ) -> Result<(Tensor<f32>, Tensor<f32>)> {
        let routed = self.route(variants, epilogues, input)?;
        let selection = variants.down_combine()?;
        let output = routed
            .activated
            .gather_quant_matmul_weighted_epilogue_with(
                &self.down_proj.packed,
                &self.down_proj.scales,
                &self.down_proj.biases,
                &routed.indices,
                &routed.weights,
                Q4_BITS,
                QUANT_GROUP,
                &epilogues.add_residual,
                &[TensorRef::new(residual)?],
                &selection,
            )?;
        Ok((output, routed.logits))
    }
}

struct RoutedMoe {
    logits: Tensor<f32>,
    weights: Tensor<f32>,
    indices: Tensor<u32>,
    activated: Tensor<f32>,
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
    cache: &'a mut Cache,
    following_norm: &'a RmsNorm<f32>,
    qk_norm_rope: &'a Kernel,
    moe_epilogues: &'a MoeEpilogues,
}

enum Cache {
    F16(KvCache<f16>),
    Bf16(KvCache<bf16>),
    F32(KvCache<f32>),
}

impl Cache {
    fn new(format: KvCacheFormat) -> Result<Self> {
        match format {
            KvCacheFormat::F16 => Ok(Self::F16(KvCache::new(
                KEY_VALUE_HEADS,
                MAX_CONTEXT,
                HEAD_DIM,
                f16::ZERO,
            )?)),
            KvCacheFormat::Bf16 => Ok(Self::Bf16(KvCache::new(
                KEY_VALUE_HEADS,
                MAX_CONTEXT,
                HEAD_DIM,
                bf16::ZERO,
            )?)),
            KvCacheFormat::F32 => Ok(Self::F32(KvCache::new(
                KEY_VALUE_HEADS,
                MAX_CONTEXT,
                HEAD_DIM,
                0.0,
            )?)),
        }
    }

    fn attention(
        &mut self,
        query: &Tensor<f32>,
        key: &Tensor<f32>,
        value: &Tensor<f32>,
        start: &Dim,
        end: &Dim,
        selection: forja_sdk::target::metal::VariantChoice<'_>,
    ) -> Result<Tensor<f32>> {
        match self {
            Self::F16(cache) => cache_attention(cache, query, key, value, start, end, selection),
            Self::Bf16(cache) => cache_attention(cache, query, key, value, start, end, selection),
            Self::F32(cache) => cache_attention(cache, query, key, value, start, end, selection),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cache_attention<C: forja_sdk::Element>(
    cache: &mut KvCache<C>,
    query: &Tensor<f32>,
    key: &Tensor<f32>,
    value: &Tensor<f32>,
    start: &Dim,
    end: &Dim,
    selection: forja_sdk::target::metal::VariantChoice<'_>,
) -> Result<Tensor<f32>> {
    cached_attention_with(
        query,
        key,
        value,
        cache,
        ATTENTION_SCALE,
        start,
        end,
        selection,
    )
}

#[derive(Clone, Copy)]
struct LayerPosition<'a> {
    activation_positions: &'a Tensor<f32>,
    sequence: u32,
    start: &'a Dim,
    end: &'a Dim,
    variants: &'a variants::Variants,
    attention_variant: &'a VariantSelection,
}

impl DecoderLayer {
    fn forward(
        &self,
        residual_kernel: &Kernel,
        norm_kernel: &Kernel,
        input: &Tensor<f32>,
        normalized_input: &Tensor<f32>,
        position: LayerPosition<'_>,
        resources: LayerResources<'_>,
    ) -> Result<(Tensor<f32>, Tensor<f32>, Tensor<f32>)> {
        let LayerResources {
            cache,
            following_norm,
            qk_norm_rope: qk_kernel,
            moe_epilogues,
        } = resources;
        let LayerPosition {
            activation_positions,
            sequence,
            start,
            end,
            variants,
            attention_variant,
        } = position;
        let (query, key, value) = self.self_attn.project_normalized(
            variants,
            normalized_input,
            sequence,
            qk_kernel,
            activation_positions,
        )?;
        let attended =
            cache.attention(&query, &key, &value, start, end, attention_variant.choice())?;
        let (hidden, normalized) = residual_norm(
            residual_kernel,
            input,
            &self.self_attn.o_proj.forward(variants, &attended)?,
            self.post_attention_layernorm.weight(),
        )?;
        let (hidden, normalized, router_logits) = if sequence == 1 {
            let (hidden, router_logits) =
                self.mlp
                    .forward_decode(variants, moe_epilogues, &normalized, &hidden)?;
            let normalized = rms_norm(norm_kernel, &hidden, following_norm.weight())?;
            (hidden, normalized, router_logits)
        } else {
            let (projected, router_logits) =
                self.mlp.forward(variants, moe_epilogues, &normalized)?;
            let (hidden, normalized) = residual_norm(
                residual_kernel,
                &hidden,
                &projected,
                following_norm.weight(),
            )?;
            (hidden, normalized, router_logits)
        };
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

/// Qwen3-Coder-30B-A3B-Instruct with a fixed 32768-token KV cache.
pub struct Qwen3Coder {
    weights: QwenWeights,
    variants: variants::Variants,
    caches: Vec<Cache>,
    residual_norm: Kernel,
    rms_norm: Kernel,
    qk_norm_rope: Kernel,
    moe_epilogues: MoeEpilogues,
    activation_positions: Tensor<f32>,
    #[cfg(target_family = "wasm")]
    replay: bool,
    #[cfg(target_family = "wasm")]
    decode: DecodeState,
    #[cfg(target_family = "wasm")]
    short_prefill: Option<ChunkedPrefill<f32>>,
    #[cfg(target_family = "wasm")]
    long_prefill: Option<ChunkedPrefill<f32>>,
}

impl Qwen3Coder {
    fn load_from_weights(
        weights: &Weights<'_>,
        layers: usize,
        config: &EngineLoadConfig,
    ) -> Result<Self> {
        let variants = variants::Variants::new(config)?;
        let weights = QwenWeights::load(weights, layers)?;
        let caches = (0..layers)
            .map(|_| Cache::new(config.kv_cache))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights,
            variants,
            caches,
            residual_norm: residual_norm_kernel(DType::F32, RMS_EPSILON)?,
            rms_norm: rms_norm_kernel(DType::F32, RMS_EPSILON)?,
            qk_norm_rope: qk_norm_rope_kernel(DType::F32, HEAD_DIM, RMS_EPSILON, ROPE_THETA)?,
            moe_epilogues: MoeEpilogues::load()?,
            activation_positions: Tensor::constant(
                &(0..MAX_CONTEXT)
                    .map(exact_position)
                    .collect::<Result<Vec<_>>>()?,
                &[MAX_CONTEXT],
            )?,
            #[cfg(target_family = "wasm")]
            replay: config.replay,
            #[cfg(target_family = "wasm")]
            decode: DecodeState::new(MAX_CONTEXT, VOCAB)?,
            #[cfg(target_family = "wasm")]
            short_prefill: Some(
                ChunkedPrefill::new(MAX_CONTEXT, SHORT_PREFILL_CHUNK)?
                    .with_tail_capacity(SHORT_PREFILL_TAIL_CAPACITY)?,
            ),
            #[cfg(target_family = "wasm")]
            long_prefill: Some(
                ChunkedPrefill::new(MAX_CONTEXT, PREFILL_CHUNK)?
                    .with_tail_chunk(PREFILL_TAIL_CHUNK)?
                    .with_tail_capacity(LONG_PREFILL_TAIL_CAPACITY)?,
            ),
        })
    }

    fn forward(
        &mut self,
        tokens: &Tensor<u32>,
        sequence: u32,
        start: &Dim,
        end: &Dim,
        taps_enabled: bool,
        attention_variant: &VariantSelection,
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
        attention_variant: &VariantSelection,
    ) -> Result<SequenceOutput> {
        let activation_positions = self.activation_positions.narrow(0, start, sequence)?;
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
                &self.rms_norm,
                &hidden,
                &normalized,
                LayerPosition {
                    activation_positions: &activation_positions,
                    sequence,
                    start,
                    end,
                    variants: &self.variants,
                    attention_variant,
                },
                LayerResources {
                    cache: &mut self.caches[index],
                    following_norm,
                    qk_norm_rope: &self.qk_norm_rope,
                    moe_epilogues: &self.moe_epilogues,
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
        let logits = self.weights.lm_head.forward(&self.variants, &normalized)?;
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
        selection: PrefillVariant<'_>,
    ) -> Result<Tensor<f32>> {
        let attention = self
            .variants
            .attention(sequence, selection.start, selection.parameter)?;
        let (logits, _, _) =
            self.forward_sequence(tokens, sequence, start, end, false, &attention)?;
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
            serving: Some(serving_info()),
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
        Self::load_from_weights(weights, layers, &config)
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
            .ok_or_else(|| forja_sdk::Error::loading("tokens exceed the 32768-token context"))?;
        #[cfg(target_family = "wasm")]
        if self.replay && sequence == 1 && !input.taps {
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
            let logits = if self.replay {
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
        let attention = self.variants.attention(sequence, input.start_pos, None)?;
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
                        forja_sdk::Error::loading("decode tokens exceed the 32768-token context")
                    })?;
                if self.prefill_supports(input.start_pos, sequence) {
                    let logits = if self.replay {
                        self.replay_prefill(&tokens, input.start_pos)?
                    } else {
                        self.lazy_prefill(&tokens, input.start_pos)?
                    };
                    self.select_token(logits, end - 1)
                } else if self.replay && sequence == 1 {
                    self.decode_selected(Some(&tokens), input.start_pos, selection)
                } else {
                    let start = input.start_pos.into();
                    let end_dim = end.into();
                    let attention = self.variants.attention(sequence, input.start_pos, None)?;
                    let logits = self
                        .forward(&tokens, sequence, &start, &end_dim, false, &attention)?
                        .logits;
                    self.select_token(logits, end - 1)
                }
            }
            None if input.start_pos < MAX_CONTEXT && self.replay => {
                self.decode_selected(None, input.start_pos, selection)
            }
            None if input.start_pos < MAX_CONTEXT => {
                let token = self.decode.token()?;
                let start = input.start_pos.into();
                let end = (input.start_pos + 1).into();
                let attention = self.variants.attention(1, input.start_pos, None)?;
                let logits = self
                    .forward(&token, 1, &start, &end, false, &attention)?
                    .logits;
                self.select_token(logits, input.start_pos)
            }
            None => Err(forja_sdk::Error::loading(
                "decode position exceeds the 32768-token context",
            )),
        }
    }

    #[cfg(target_family = "wasm")]
    fn begin_decode(&mut self, input: forja_sdk::BeginDecodeInput) -> Result<Tensor<f32>> {
        if !self.replay || input.start_pos >= MAX_CONTEXT {
            return Err(forja_sdk::Error::loading(
                "constrained decode requires replay within context",
            ));
        }
        if let Some(tokens) = input.tokens.as_ref()
            && tokens.shape() != [1]
        {
            return Err(forja_sdk::Error::loading(
                "constrained decode accepts exactly one input token",
            ));
        }
        self.replay_decode(
            input.tokens.as_ref(),
            input.start_pos,
            DecodeSelection::None,
        )
    }

    #[cfg(target_family = "wasm")]
    fn finish_decode(
        &mut self,
        logits: &Tensor<f32>,
        start_pos: u32,
        sampling: forja_sdk::SamplingParams,
        allow: &[u32],
    ) -> Result<forja_sdk::DecodeOutput> {
        self.decode.select(sampling)?;
        self.decode.write_allow(allow)?;
        if self.decode.masked_graph().is_none() {
            let graph = self.capture_masked_selection(logits, start_pos)?;
            self.decode.set_masked_graph(graph);
        }
        let graph = self
            .decode
            .masked_graph()
            .ok_or_else(|| forja_sdk::Error::loading("masked selection graph was not captured"))?;
        graph.replay(&[start_pos])?;
        let next_pos = start_pos
            .checked_add(1)
            .ok_or_else(|| forja_sdk::Error::loading("decode position overflowed"))?;
        let token = graph.result().alias()?;
        self.decode.advance(next_pos);
        Ok(forja_sdk::DecodeOutput {
            logits: logits.alias()?,
            token,
        })
    }

    #[cfg(target_family = "wasm")]
    fn rewind_decode(&mut self, next_pos: u32) -> Result<()> {
        self.decode.rewind(next_pos)
    }
}

#[cfg(target_family = "wasm")]
impl Qwen3Coder {
    fn prefill_supports(&self, start: u32, sequence: u32) -> bool {
        self.prefill_for(sequence)
            .is_some_and(|prefill| prefill.supports(start, sequence))
    }

    fn replay_prefill(&mut self, tokens: &Tensor<u32>, start: u32) -> Result<Tensor<f32>> {
        let [sequence] = tokens
            .shape()
            .try_into()
            .map_err(|_| forja_sdk::Error::loading("prefill tokens must have rank one"))?;
        let mut prefill = self
            .prefill_for_mut(sequence)
            .take()
            .ok_or_else(|| forja_sdk::Error::loading("prefill state is unavailable"))?;
        let result = prefill.replay(
            tokens,
            start,
            |tokens, sequence, last, start, end, execution| {
                let variant = match execution {
                    PrefillExecution::Capture(parameter) => PrefillVariant {
                        start: 0,
                        parameter: Some(parameter),
                    },
                    PrefillExecution::Lazy(start) => PrefillVariant {
                        start,
                        parameter: None,
                    },
                };
                self.forward_prefill_chunk(tokens, sequence, last, start, end, variant)
            },
        );
        *self.prefill_for_mut(sequence) = Some(prefill);
        result
    }

    fn lazy_prefill(&mut self, tokens: &Tensor<u32>, start: u32) -> Result<Tensor<f32>> {
        let [sequence] = tokens
            .shape()
            .try_into()
            .map_err(|_| forja_sdk::Error::loading("prefill tokens must have rank one"))?;
        let prefill = self
            .prefill_for_mut(sequence)
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
        *self.prefill_for_mut(sequence) = Some(prefill);
        result
    }

    fn prefill_for(&self, sequence: u32) -> Option<&ChunkedPrefill<f32>> {
        if sequence < PREFILL_CHUNK {
            self.short_prefill.as_ref()
        } else {
            self.long_prefill.as_ref()
        }
    }

    fn prefill_for_mut(&mut self, sequence: u32) -> &mut Option<ChunkedPrefill<f32>> {
        if sequence < PREFILL_CHUNK {
            &mut self.short_prefill
        } else {
            &mut self.long_prefill
        }
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
        let sample_variant = self.variants.sample()?;
        forja_sdk::capture(&[&position], || {
            let attention = self.variants.attention(1, start_pos, Some(&position))?;
            let logits = self
                .forward(&token, 1, &start_dim, &end, false, &attention)?
                .logits;
            let selected = match selection {
                DecodeSelection::None => None,
                DecodeSelection::Greedy => Some(logits.reshape(&[1, VOCAB])?.argmax()?),
                DecodeSelection::Sampled => Some(logits.reshape(&[1, VOCAB])?.sample_with(
                    &sampling,
                    start.clone(),
                    &sample_variant,
                )?),
            };
            if let Some(selected) = selected {
                selected.copy_into(&mut feedback)?;
                selected.copy_into(&mut output_tokens.narrow(0, start, 1)?)?;
            }
            Ok(logits)
        })
    }

    #[cfg(target_family = "wasm")]
    fn capture_masked_selection(
        &mut self,
        logits: &Tensor<f32>,
        start_pos: u32,
    ) -> Result<forja_sdk::Graph<Tensor<u32>>> {
        let position = forja_sdk::Param::new(0..=MAX_CONTEXT - 1)?;
        let start = position.at(start_pos);
        let sampling = self.decode.sampling()?;
        let allow = self.decode.allow()?;
        let mut feedback = self.decode.token()?;
        let output_tokens = self.decode.output_tokens()?;
        forja_sdk::capture(&[&position], || {
            let selected =
                logits
                    .reshape(&[1, VOCAB])?
                    .sample_masked(&sampling, &allow, start.clone())?;
            selected.copy_into(&mut feedback)?;
            selected.copy_into(&mut output_tokens.narrow(0, start, 1)?)?;
            Ok(selected)
        })
    }

    fn decode_selected(
        &mut self,
        tokens: Option<&Tensor<u32>>,
        start_pos: u32,
        selection: DecodeSelection,
    ) -> Result<forja_sdk::DecodeOutput> {
        let logits = self.replay_decode(tokens, start_pos, selection)?;
        let next_pos = start_pos
            .checked_add(1)
            .ok_or_else(|| forja_sdk::Error::loading("decode position overflowed"))?;
        self.decode.advance(next_pos);
        Ok(forja_sdk::DecodeOutput {
            logits,
            token: self.decode.selected(start_pos)?,
        })
    }

    fn select_token(&mut self, logits: Tensor<f32>, slot: u32) -> Result<forja_sdk::DecodeOutput> {
        let sample_variant = self.variants.sample()?;
        let selected = logits.reshape(&[1, VOCAB])?.sample_with(
            &self.decode.sampling()?,
            slot,
            &sample_variant,
        )?;
        let token = self.decode.store_selected(&selected, slot)?;
        self.decode.advance(
            slot.checked_add(1)
                .ok_or_else(|| forja_sdk::Error::loading("decode position overflowed"))?,
        );
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
        let serving = info.serving.unwrap();
        assert_eq!(serving.page_tokens, 64);
        assert_eq!(serving.planes.len(), 96);
        assert_eq!(serving.planes[0].elements_per_token, 512);
        assert_eq!(serving.active_lanes, 8);
        assert_eq!(serving.max_context, MAX_CONTEXT);
    }

    #[test]
    fn builds_exact_positions_past_u16() {
        assert_eq!(
            exact_position(u32::from(u16::MAX) + 1).unwrap().to_bits(),
            65_536.0_f32.to_bits()
        );
        assert_eq!(
            exact_position(1 << f32::MANTISSA_DIGITS).unwrap().to_bits(),
            16_777_216.0_f32.to_bits()
        );
        assert!(exact_position((1 << f32::MANTISSA_DIGITS) + 1).is_err());
    }
}
