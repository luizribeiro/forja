//! Qwen3-0.6B inference engine.

mod slots;
mod tuning;

#[cfg(target_family = "wasm")]
use forja_sdk::nn::blocks::{ChunkedPrefill, DEFAULT_PREFILL_CHUNK, DecodeSelection, DecodeState};
use forja_sdk::{
    DType, Dim, Engine, EngineInfo, FloatElement, Load, Param, Result, StepInput, StepOutput,
    Tensor, Weights, bf16, export_engine,
    kernel::{Kernel, TensorRef},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, WeightElement,
        blocks::{KvCache, Taps, cached_attention_with, qk_norm_rope, residual_norm},
    },
    target::metal::{Variant, VariantChoice, VariantRule},
};

/// Vocabulary size reported by Qwen3-0.6B.
pub const VOCAB: u32 = 151_936;
/// Hidden width of Qwen3-0.6B.
pub const HIDDEN: u32 = 1_024;
/// Number of decoder layers.
pub const LAYERS: usize = 28;
/// Largest context accepted by this engine.
pub const MAX_CONTEXT: u32 = 4_096;

const QUERY_HEADS: u32 = 16;
const KEY_VALUE_HEADS: u32 = 8;
const HEAD_DIM: u32 = 128;
const INTERMEDIATE: u32 = 3_072;
const RMS_EPSILON: f32 = 1.0e-6;
const ROPE_THETA: f32 = 1.0e6;
const ATTENTION_SCALE: f32 = 0.088_388_35;
#[cfg(target_family = "wasm")]
const REPLAY_DECODE: bool = !cfg!(feature = "no-replay");

#[derive(Clone, Copy)]
struct Config;

/// Activation precision supported by the Qwen3 engine.
pub trait Activation: WeightElement + FloatElement {
    /// Additive identity used to initialize the KV cache.
    const ZERO: Self;

    /// Kernel signature type for this activation storage.
    const KERNEL_DTYPE: DType;

    /// Converts a context position to the activation representation.
    fn from_position(position: u16) -> Self;

    /// Selects the model's bf16 storage representation.
    fn linear_config(input: u32, output: u32) -> LinearConfig;

    /// Selects the model's bf16 storage representation.
    fn norm_config(hidden: u32, epsilon: f32) -> RmsNormConfig;

    /// Selects the model's bf16 storage representation.
    fn embedding_config(vocab: u32, hidden: u32) -> EmbeddingConfig;

    /// Converts an engine result to the component's f32 output type.
    ///
    /// # Errors
    ///
    /// Returns an error when conversion or materialization fails.
    fn output(tensor: Tensor<Self>) -> Result<Tensor<f32>>;
}

impl Activation for f32 {
    const ZERO: Self = 0.0;
    const KERNEL_DTYPE: DType = DType::F32;

    fn from_position(position: u16) -> Self {
        Self::from(position)
    }

    fn linear_config(input: u32, output: u32) -> LinearConfig {
        LinearConfig::promoted_bf16(input, output)
    }

    fn norm_config(hidden: u32, epsilon: f32) -> RmsNormConfig {
        RmsNormConfig::promoted_bf16(hidden, epsilon)
    }

    fn embedding_config(vocab: u32, hidden: u32) -> EmbeddingConfig {
        EmbeddingConfig::promoted_bf16(vocab, hidden)
    }

    fn output(tensor: Tensor<Self>) -> Result<Tensor<f32>> {
        Ok(tensor)
    }
}

impl Activation for bf16 {
    const ZERO: Self = bf16::ZERO;
    const KERNEL_DTYPE: DType = DType::BF16;

    fn from_position(position: u16) -> Self {
        Self::from_f32(f32::from(position))
    }

    fn linear_config(input: u32, output: u32) -> LinearConfig {
        LinearConfig::new(input, output)
    }

    fn norm_config(hidden: u32, epsilon: f32) -> RmsNormConfig {
        RmsNormConfig::new(hidden, epsilon)
    }

    fn embedding_config(vocab: u32, hidden: u32) -> EmbeddingConfig {
        EmbeddingConfig::new(vocab, hidden)
    }

    fn output(tensor: Tensor<Self>) -> Result<Tensor<f32>> {
        tensor.to_dtype()
    }
}

fn variant(name: &str) -> Result<Variant> {
    Variant::new(name)
}

fn matmul_variant<T: Activation>(rows: u32, inner: u32, columns: u32) -> Result<Variant> {
    if rows == 1 {
        return variant("matmul.gemv-transposed");
    }
    let large = u64::from(rows)
        .checked_mul(u64::from(columns))
        .is_some_and(|elements| elements >= 1 << 20);
    let half = matches!(T::KERNEL_DTYPE, DType::F16 | DType::BF16);
    let name = if half && large && u64::from(rows.max(columns)) * 2 > u64::from(inner) {
        "matmul.steel-64x64x16-1x2"
    } else if half {
        "matmul.steel-64x32x32-2x2"
    } else if large {
        "matmul.steel-64x64x16-2x2"
    } else {
        "matmul.steel-32x64x16-1x2"
    };
    variant(name)
}

fn linear<T: Activation>(
    projection: &Linear<T>,
    input: &Tensor<T>,
    rows: u32,
    inner: u32,
    columns: u32,
) -> Result<Tensor<T>> {
    let selection = matmul_variant::<T>(rows, inner, columns)?;
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
    let name = if sequence == 1 {
        match start
            .checked_add(1)
            .ok_or_else(|| forja_sdk::Error::loading("attention cache length overflowed"))?
        {
            ..512 => "sdpa.decomposed",
            512..1024 => "sdpa.vector-single-pass",
            _ => "sdpa.vector-two-pass",
        }
    } else if sequence >= 512 && start == 0 {
        "sdpa.steel"
    } else {
        "sdpa.decomposed"
    };
    Ok(AttentionVariant::Fixed(variant(name)?))
}

#[derive(Load)]
#[load(config = Config)]
struct Attention<T: Activation> {
    #[load(prefix, config = T::linear_config(HIDDEN, QUERY_HEADS * HEAD_DIM))]
    q_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(HIDDEN, KEY_VALUE_HEADS * HEAD_DIM))]
    k_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(HIDDEN, KEY_VALUE_HEADS * HEAD_DIM))]
    v_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(QUERY_HEADS * HEAD_DIM, HIDDEN))]
    o_proj: Linear<T>,
    #[load(prefix, config = T::norm_config(HEAD_DIM, RMS_EPSILON))]
    q_norm: RmsNorm<T>,
    #[load(prefix, config = T::norm_config(HEAD_DIM, RMS_EPSILON))]
    k_norm: RmsNorm<T>,
}

#[derive(Load)]
#[load(config = Config)]
#[allow(clippy::struct_field_names)]
struct Mlp<T: Activation> {
    #[load(prefix, config = T::linear_config(HIDDEN, INTERMEDIATE))]
    gate_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(HIDDEN, INTERMEDIATE))]
    up_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(INTERMEDIATE, HIDDEN))]
    down_proj: Linear<T>,
}

impl<T: Activation> Attention<T> {
    fn project(&self, input: &Tensor<T>, sequence: u32) -> Result<[Tensor<T>; 3]> {
        let query = linear(
            &self.q_proj,
            input,
            sequence,
            HIDDEN,
            QUERY_HEADS * HEAD_DIM,
        )?
        .reshape(&[sequence, QUERY_HEADS, HEAD_DIM])?;
        let key = linear(
            &self.k_proj,
            input,
            sequence,
            HIDDEN,
            KEY_VALUE_HEADS * HEAD_DIM,
        )?
        .reshape(&[sequence, KEY_VALUE_HEADS, HEAD_DIM])?;
        let value = linear(
            &self.v_proj,
            input,
            sequence,
            HIDDEN,
            KEY_VALUE_HEADS * HEAD_DIM,
        )?;
        Ok([query, key, value])
    }
}

impl<T: Activation> Mlp<T> {
    fn forward(
        &self,
        kernels: &FusedKernels,
        input: &Tensor<T>,
        sequence: u32,
    ) -> Result<Tensor<T>> {
        let gate = linear(&self.gate_proj, input, sequence, HIDDEN, INTERMEDIATE)?;
        let up = linear(&self.up_proj, input, sequence, HIDDEN, INTERMEDIATE)?;
        let activated = if kernels.has_silu_mul() {
            run_silu_mul(kernels.silu_mul()?, &gate, &up)?
        } else {
            gate.silu_mul(&up)?
        };
        linear(&self.down_proj, &activated, sequence, INTERMEDIATE, HIDDEN)
    }
}

#[derive(Load)]
#[load(config = Config)]
struct DecoderLayer<T: Activation> {
    #[load(prefix, config = T::norm_config(HIDDEN, RMS_EPSILON))]
    input_layernorm: RmsNorm<T>,
    #[load(prefix)]
    self_attn: Attention<T>,
    #[load(prefix, config = T::norm_config(HIDDEN, RMS_EPSILON))]
    post_attention_layernorm: RmsNorm<T>,
    #[load(prefix)]
    mlp: Mlp<T>,
}

#[derive(Load)]
#[load(config = Config)]
struct Model<T: Activation> {
    #[load(prefix, config = T::embedding_config(VOCAB, HIDDEN))]
    embed_tokens: Embedding<T>,
    #[load(prefix, count = LAYERS)]
    layers: Vec<DecoderLayer<T>>,
    #[load(prefix, config = T::norm_config(HIDDEN, RMS_EPSILON))]
    norm: RmsNorm<T>,
}

#[derive(Load)]
#[load(config = Config)]
struct QwenWeights<T: Activation> {
    #[load(prefix)]
    model: Model<T>,
}

struct FusedKernels {
    silu_mul: Option<Kernel>,
    qk_norm_rope: Option<Kernel>,
    residual_norm: Option<Kernel>,
    final_norm: Option<Kernel>,
}

impl FusedKernels {
    fn load<T: Activation>(slots: slots::Slots) -> Result<Self> {
        let dtype = T::KERNEL_DTYPE;
        Ok(Self {
            silu_mul: load_kernel(slots.silu_mul.is_tuned(), || {
                tuning::silu_mul::kernel(dtype)
            })?,
            qk_norm_rope: load_kernel(slots.qk_norm_rope.is_tuned(), || {
                tuning::qk_norm_rope::kernel(dtype)
            })?,
            residual_norm: load_kernel(slots.residual_norm.is_tuned(), || {
                tuning::residual_norm::kernel(dtype)
            })?,
            final_norm: load_kernel(slots.final_norm.is_tuned(), || {
                tuning::final_norm::kernel(dtype)
            })?,
        })
    }

    const fn has_silu_mul(&self) -> bool {
        self.silu_mul.is_some()
    }

    const fn has_qk_norm_rope(&self) -> bool {
        self.qk_norm_rope.is_some()
    }

    const fn has_residual_norm(&self) -> bool {
        self.residual_norm.is_some()
    }

    const fn has_final_norm(&self) -> bool {
        self.final_norm.is_some()
    }

    fn silu_mul(&self) -> Result<&Kernel> {
        required_kernel(self.silu_mul.as_ref(), "silu_mul")
    }

    fn qk_norm_rope(&self) -> Result<&Kernel> {
        required_kernel(self.qk_norm_rope.as_ref(), "qk_norm_rope")
    }

    fn residual_norm(&self) -> Result<&Kernel> {
        required_kernel(self.residual_norm.as_ref(), "residual_norm")
    }

    fn final_norm(&self) -> Result<&Kernel> {
        required_kernel(self.final_norm.as_ref(), "final_norm")
    }
}

fn load_kernel(enabled: bool, build: impl FnOnce() -> Result<Kernel>) -> Result<Option<Kernel>> {
    enabled.then(build).transpose()
}

fn required_kernel<'a>(kernel: Option<&'a Kernel>, name: &str) -> Result<&'a Kernel> {
    kernel.ok_or_else(|| forja_sdk::Error::loading(format!("fused kernel `{name}` is unavailable")))
}

struct LayerResources<'a, T: Activation> {
    cache: &'a mut KvCache<T>,
    following_norm: Option<&'a RmsNorm<T>>,
}

#[derive(Clone, Copy)]
struct LayerPosition<'a, T: Activation> {
    positions: &'a Tensor<u32>,
    activation_positions: Option<&'a Tensor<T>>,
    sequence: u32,
    start: &'a Dim,
    end: &'a Dim,
    attention: &'a AttentionVariant,
}

enum AttentionVariant {
    Fixed(Variant),
    Rule(VariantRule),
}

#[cfg(target_family = "wasm")]
#[derive(Clone, Copy)]
struct PrefillVariant<'a> {
    start: u32,
    parameter: Option<&'a Param>,
}

impl AttentionVariant {
    fn choice(&self) -> VariantChoice<'_> {
        match self {
            Self::Fixed(variant) => variant.into(),
            Self::Rule(rule) => rule.into(),
        }
    }
}

impl<T: Activation> DecoderLayer<T> {
    fn forward(
        &self,
        kernels: &FusedKernels,
        input: &Tensor<T>,
        normalized_input: &Tensor<T>,
        position: LayerPosition<'_, T>,
        resources: LayerResources<'_, T>,
    ) -> Result<(Tensor<T>, Option<Tensor<T>>)> {
        let LayerResources {
            cache,
            following_norm,
        } = resources;
        let LayerPosition {
            positions,
            activation_positions,
            sequence,
            start,
            end,
            attention,
        } = position;
        let [query_projection, key_projection, value_projection] =
            self.self_attn.project(normalized_input, sequence)?;
        let (query, key) = if kernels.has_qk_norm_rope() {
            let program_positions = activation_positions.ok_or_else(|| {
                forja_sdk::Error::loading("fused rotary positions are unavailable")
            })?;
            (
                qk_norm_rope(
                    kernels.qk_norm_rope()?,
                    &query_projection,
                    self.self_attn.q_norm.weight(),
                    program_positions,
                )?,
                qk_norm_rope(
                    kernels.qk_norm_rope()?,
                    &key_projection,
                    self.self_attn.k_norm.weight(),
                    program_positions,
                )?,
            )
        } else {
            (
                self.self_attn
                    .q_norm
                    .forward(&query_projection)?
                    .rope(positions, ROPE_THETA)?,
                self.self_attn
                    .k_norm
                    .forward(&key_projection)?
                    .rope(positions, ROPE_THETA)?,
            )
        };
        let value = value_projection.reshape(&[sequence, KEY_VALUE_HEADS, HEAD_DIM])?;
        let attended = cached_attention_with(
            &query,
            &key,
            &value,
            cache,
            ATTENTION_SCALE,
            start,
            end,
            attention.choice(),
        )?;
        let attention = linear(
            &self.self_attn.o_proj,
            &attended,
            sequence,
            QUERY_HEADS * HEAD_DIM,
            HIDDEN,
        )?;
        let (hidden, normalized) = if kernels.has_residual_norm() {
            residual_norm(
                kernels.residual_norm()?,
                input,
                &attention,
                self.post_attention_layernorm.weight(),
            )?
        } else {
            let hidden = (input + &attention)?;
            let normalized = self.post_attention_layernorm.forward(&hidden)?;
            (hidden, normalized)
        };
        let projected = self.mlp.forward(kernels, &normalized, sequence)?;
        if kernels.has_residual_norm()
            && let Some(norm) = following_norm
        {
            let (residual, normalized) =
                residual_norm(kernels.residual_norm()?, &hidden, &projected, norm.weight())?;
            Ok((residual, Some(normalized)))
        } else {
            let residual = (&hidden + &projected)?;
            let normalized = following_norm
                .map(|norm| norm.forward(&residual))
                .transpose()?;
            Ok((residual, normalized))
        }
    }
}

fn run_silu_mul<T: Activation>(
    kernel: &Kernel,
    gate: &Tensor<T>,
    up: &Tensor<T>,
) -> Result<Tensor<T>> {
    let inputs = [TensorRef::new(gate)?, TensorRef::new(up)?];
    let [output] = forja_sdk::kernel::run::<T, 1>(kernel, &inputs)?;
    Ok(output)
}

fn run_final_norm<T: Activation>(
    kernel: &Kernel,
    input: &Tensor<T>,
    weight: &Tensor<T>,
) -> Result<Tensor<T>> {
    let weight = weight.broadcast_as(input.shape())?;
    let inputs = [TensorRef::new(input)?, TensorRef::new(&weight)?];
    let [output] = forja_sdk::kernel::run::<T, 1>(kernel, &inputs)?;
    Ok(output)
}

/// Qwen3-0.6B with a fixed 4096-token KV cache.
pub struct Qwen3<T: Activation = f32> {
    weights: QwenWeights<T>,
    caches: Vec<KvCache<T>>,
    kernels: FusedKernels,
    positions: Tensor<u32>,
    activation_positions: Option<Tensor<T>>,
    #[cfg(target_family = "wasm")]
    decode: DecodeState,
    #[cfg(target_family = "wasm")]
    prefill: Option<ChunkedPrefill<f32>>,
}

impl<T: Activation> Qwen3<T> {
    fn load_from_weights(
        weights: &Weights<'_>,
        config: &forja_sdk::EngineLoadConfig,
    ) -> Result<Self> {
        let slots = slots::Slots::new(config)?;
        let weights = QwenWeights::<_>::load(weights, &Config)?;
        let caches = (0..LAYERS)
            .map(|_| KvCache::new(KEY_VALUE_HEADS, MAX_CONTEXT, HEAD_DIM, T::ZERO))
            .collect::<Result<Vec<_>>>()?;
        let kernels = FusedKernels::load::<T>(slots)?;
        let positions = Tensor::constant(&(0..MAX_CONTEXT).collect::<Vec<_>>(), &[MAX_CONTEXT])?;
        let activation_positions = kernels
            .has_qk_norm_rope()
            .then(|| activation_position_table::<T>())
            .transpose()?;
        Ok(Self {
            weights,
            caches,
            kernels,
            positions,
            activation_positions,
            #[cfg(target_family = "wasm")]
            decode: DecodeState::new(MAX_CONTEXT)?,
            #[cfg(target_family = "wasm")]
            prefill: Some(ChunkedPrefill::new(MAX_CONTEXT, DEFAULT_PREFILL_CHUNK)?),
        })
    }

    fn output(tensor: Tensor<T>) -> Result<Tensor<f32>> {
        T::output(tensor)
    }

    /// Runs the embedding and first decoder layer for native differential checks.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid token or operation inputs, or failed execution.
    pub fn first_layer(&mut self, tokens: &Tensor<u32>) -> Result<Tensor<T>> {
        let sequence = tokens
            .shape()
            .first()
            .copied()
            .ok_or_else(|| forja_sdk::Error::loading("tokens must have rank one"))?;
        let positions = self.positions.narrow(0, 0, sequence)?;
        let program_positions = self
            .activation_positions
            .as_ref()
            .map(|positions| positions.narrow(0, 0, sequence))
            .transpose()?;
        let hidden = self.weights.model.embed_tokens.forward(tokens)?;
        let normalized = self.weights.model.layers[0]
            .input_layernorm
            .forward(&hidden)?;
        let start = 0.into();
        let end = sequence.into();
        let attention = attention_variant(sequence, 0, None)?;
        self.weights.model.layers[0]
            .forward(
                &self.kernels,
                &hidden,
                &normalized,
                LayerPosition {
                    positions: &positions,
                    activation_positions: program_positions.as_ref(),
                    sequence,
                    start: &start,
                    end: &end,
                    attention: &attention,
                },
                LayerResources {
                    cache: &mut self.caches[0],
                    following_norm: None,
                },
            )?
            .0
            .contiguous()
    }
}

#[cfg(all(target_family = "wasm", feature = "bf16"))]
type ExportedQwen3 = Qwen3<bf16>;
#[cfg(not(all(target_family = "wasm", feature = "bf16")))]
type ExportedQwen3 = Qwen3<f32>;

#[export_engine]
impl Engine for ExportedQwen3 {
    /// Tap `i` is decoder layer `i`; tap 28 includes the final model norm.
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: VOCAB,
            max_context: MAX_CONTEXT,
            tap_layers: (1..=28).collect(),
            router_layers: Vec::new(),
        }
    }

    fn load(weights: &Weights<'_>, config: forja_sdk::EngineLoadConfig) -> Result<Self> {
        Self::load_from_weights(weights, &config)
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
        let end_pos = input
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
        let end = end_pos.into();
        let attention = attention_variant(sequence, input.start_pos, None)?;
        self.forward(
            &input.tokens,
            sequence,
            &start,
            &end,
            &attention,
            input.taps,
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
                        .forward(&tokens, sequence, &start, &end_dim, &attention, false)?
                        .logits;
                    self.select_token(logits, end - 1)
                }
            }
            None if input.start_pos < MAX_CONTEXT && REPLAY_DECODE => {
                self.decode_selected(None, input.start_pos, selection)
            }
            None if input.start_pos < MAX_CONTEXT => {
                let token = self.decode.token()?;
                let end = input.start_pos + 1;
                let start_dim = input.start_pos.into();
                let end_dim = end.into();
                let attention = attention_variant(1, input.start_pos, None)?;
                let logits = self
                    .forward(&token, 1, &start_dim, &end_dim, &attention, false)?
                    .logits;
                self.select_token(logits, input.start_pos)
            }
            None => Err(forja_sdk::Error::loading(
                "decode position exceeds the 4096-token context",
            )),
        }
    }
}

impl<T: Activation> Qwen3<T> {
    fn forward(
        &mut self,
        tokens: &Tensor<u32>,
        sequence: u32,
        start: &Dim,
        end: &Dim,
        attention: &AttentionVariant,
        taps_enabled: bool,
    ) -> Result<StepOutput> {
        let (logits, taps) =
            self.forward_sequence(tokens, sequence, start, end, attention, taps_enabled)?;
        Ok(StepOutput {
            logits: Self::last_logits(
                &logits,
                sequence
                    .checked_sub(1)
                    .ok_or_else(|| forja_sdk::Error::loading("tokens cannot be empty"))?,
            )?,
            taps,
            router_logits: Vec::new(),
        })
    }

    fn forward_sequence(
        &mut self,
        tokens: &Tensor<u32>,
        sequence: u32,
        start: &Dim,
        end: &Dim,
        attention: &AttentionVariant,
        taps_enabled: bool,
    ) -> Result<(Tensor<T>, Vec<Tensor<f32>>)> {
        let positions = self.positions.narrow(0, start, sequence)?;
        let program_positions = self
            .activation_positions
            .as_ref()
            .map(|positions| positions.narrow(0, start, sequence))
            .transpose()?;
        let mut hidden = self.weights.model.embed_tokens.forward(tokens)?;
        let mut normalized = self.weights.model.layers[0]
            .input_layernorm
            .forward(&hidden)?;
        let mut taps = Taps::new(taps_enabled, LAYERS);
        for index in 0..LAYERS {
            let layer = &self.weights.model.layers[index];
            let following_norm = self
                .weights
                .model
                .layers
                .get(index + 1)
                .map(|layer| &layer.input_layernorm);
            let (next_hidden, next_normalized) = layer.forward(
                &self.kernels,
                &hidden,
                &normalized,
                LayerPosition {
                    positions: &positions,
                    activation_positions: program_positions.as_ref(),
                    sequence,
                    start,
                    end,
                    attention,
                },
                LayerResources {
                    cache: &mut self.caches[index],
                    following_norm,
                },
            )?;
            hidden = next_hidden;
            if taps.enabled() && index + 1 < LAYERS {
                taps.push(Self::output(hidden.contiguous()?)?);
            }
            if let Some(value) = next_normalized {
                normalized = value;
            }
        }
        hidden = if self.kernels.has_final_norm() {
            run_final_norm(
                self.kernels.final_norm()?,
                &hidden,
                self.weights.model.norm.weight(),
            )?
        } else {
            self.weights.model.norm.forward(&hidden)?
        };
        if taps.enabled() {
            taps.push(Self::output(hidden.contiguous()?)?);
        }
        let selection = matmul_variant::<T>(sequence, HIDDEN, VOCAB)?;
        let logits = self
            .weights
            .model
            .embed_tokens
            .project_with(&hidden, &selection)?;
        Ok((logits, taps.finish()))
    }

    fn last_logits(logits: &Tensor<T>, last: impl Into<Dim>) -> Result<Tensor<f32>> {
        Self::output(logits.narrow(0, last, 1)?.reshape(&[VOCAB])?.contiguous()?)
    }

    #[cfg(target_family = "wasm")]
    fn forward_prefill_chunk(
        &mut self,
        tokens: &Tensor<u32>,
        sequence: u32,
        last: &Dim,
        start: &Dim,
        end: &Dim,
        variant: PrefillVariant<'_>,
    ) -> Result<Tensor<f32>> {
        let attention = attention_variant(sequence, variant.start, variant.parameter)?;
        let (logits, _) = self.forward_sequence(tokens, sequence, start, end, &attention, false)?;
        Self::last_logits(&logits, last)
    }

    #[cfg(target_family = "wasm")]
    fn prefill_supports(&self, start: u32, sequence: u32) -> bool {
        self.prefill
            .as_ref()
            .is_some_and(|prefill| prefill.supports(start, sequence))
    }

    #[cfg(target_family = "wasm")]
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

    #[cfg(target_family = "wasm")]
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

    #[cfg(target_family = "wasm")]
    fn replay_decode(
        &mut self,
        tokens: Option<&Tensor<u32>>,
        start_pos: u32,
        selection: DecodeSelection,
    ) -> Result<Tensor<f32>> {
        if let Some(tokens) = tokens {
            self.decode.write_token(tokens)?;
        }
        let missing = self.decode.graph(selection).is_none();
        if missing {
            let graph = self.capture_decode(start_pos, selection)?;
            self.decode.set_graph(selection, graph);
        }
        let graph = self
            .decode_graph(selection)
            .ok_or_else(|| forja_sdk::Error::loading("decode graph was not captured"))?;
        graph.replay(&[start_pos])?;
        graph.result().alias()
    }

    #[cfg(target_family = "wasm")]
    fn decode_graph(&self, selection: DecodeSelection) -> Option<&forja_sdk::Graph<Tensor<f32>>> {
        self.decode.graph(selection)
    }

    #[cfg(target_family = "wasm")]
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
        let attention = attention_variant(1, start_pos, Some(&position))?;
        forja_sdk::capture(&[&position], || {
            let logits = self
                .forward(&token, 1, &start_dim, &end, &attention, false)?
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

    #[cfg(target_family = "wasm")]
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

    #[cfg(target_family = "wasm")]
    fn select_token(&mut self, logits: Tensor<f32>, slot: u32) -> Result<forja_sdk::DecodeOutput> {
        let selected = logits
            .reshape(&[1, VOCAB])?
            .sample(&self.decode.sampling()?, slot)?;
        let output = self.decode.store_selected(&selected, slot)?;
        Ok(forja_sdk::DecodeOutput {
            logits,
            token: output,
        })
    }
}

fn activation_position_table<T: Activation>() -> Result<Tensor<T>> {
    let values = (0..MAX_CONTEXT)
        .map(|position| {
            u16::try_from(position)
                .map(T::from_position)
                .map_err(|_| forja_sdk::Error::loading("position exceeds activation range"))
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::constant(&values, &[MAX_CONTEXT])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixed_name(selection: AttentionVariant) -> String {
        match selection {
            AttentionVariant::Fixed(variant) => variant.name().to_owned(),
            AttentionVariant::Rule(_) => panic!("expected a fixed attention variant"),
        }
    }

    #[test]
    fn preserves_dense_matmul_routing() {
        assert_eq!(
            matmul_variant::<bf16>(1, HIDDEN, INTERMEDIATE)
                .unwrap()
                .name(),
            "matmul.gemv-transposed"
        );
        assert_eq!(
            matmul_variant::<bf16>(512, HIDDEN, INTERMEDIATE)
                .unwrap()
                .name(),
            "matmul.steel-64x64x16-1x2"
        );
        assert_eq!(
            matmul_variant::<bf16>(512, HIDDEN, KEY_VALUE_HEADS * HEAD_DIM)
                .unwrap()
                .name(),
            "matmul.steel-64x32x32-2x2"
        );
        assert_eq!(
            matmul_variant::<f32>(16, HIDDEN, INTERMEDIATE)
                .unwrap()
                .name(),
            "matmul.steel-32x64x16-1x2"
        );
    }

    #[test]
    fn preserves_attention_boundaries() {
        assert_eq!(
            fixed_name(attention_variant(1, 510, None).unwrap()),
            "sdpa.decomposed"
        );
        assert_eq!(
            fixed_name(attention_variant(1, 511, None).unwrap()),
            "sdpa.vector-single-pass"
        );
        assert_eq!(
            fixed_name(attention_variant(1, 1023, None).unwrap()),
            "sdpa.vector-two-pass"
        );
        assert_eq!(
            fixed_name(attention_variant(512, 0, None).unwrap()),
            "sdpa.steel"
        );
        assert_eq!(
            fixed_name(attention_variant(512, 1, None).unwrap()),
            "sdpa.decomposed"
        );
        let parameter = Param::new(0..=MAX_CONTEXT - 1).unwrap();
        assert!(matches!(
            attention_variant(1, 0, Some(&parameter)).unwrap(),
            AttentionVariant::Rule(_)
        ));
    }
}
