//! Qwen3-0.6B inference engine.

#[cfg(any(
    all(
        not(feature = "all-fusions"),
        feature = "residual-norm-only",
        any(
            feature = "qk-norm-rope-only",
            feature = "silu-mul-only",
            feature = "final-norm-only"
        )
    ),
    all(
        not(feature = "all-fusions"),
        feature = "qk-norm-rope-only",
        any(feature = "silu-mul-only", feature = "final-norm-only")
    ),
    all(
        not(feature = "all-fusions"),
        feature = "silu-mul-only",
        feature = "final-norm-only"
    )
))]
compile_error!("select at most one fusion profile");

#[cfg(target_family = "wasm")]
use forja_sdk::nn::blocks::{DecodeSelection, DecodeState};
use forja_sdk::{
    DType, Dim, Engine, EngineInfo, FloatElement, Load, Result, StepInput, StepOutput, Tensor,
    Weights, bf16, export_engine,
    kernel::{Kernel, TensorRef},
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, WeightElement,
        blocks::KvCache, ops::sdpa,
    },
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
const FUSE_RESIDUAL_NORM: bool = cfg!(any(feature = "residual-norm-only", feature = "all-fusions"));
const FUSE_QK_NORM_ROPE: bool = cfg!(any(feature = "qk-norm-rope-only", feature = "all-fusions"));
const FUSE_SILU_MUL: bool = cfg!(any(feature = "silu-mul-only", feature = "all-fusions"));
const FUSE_FINAL_NORM: bool = cfg!(any(feature = "final-norm-only", feature = "all-fusions"));
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
struct Mlp<T: Activation> {
    #[load(prefix, config = T::linear_config(HIDDEN, INTERMEDIATE))]
    gate_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(HIDDEN, INTERMEDIATE))]
    up_proj: Linear<T>,
    #[load(prefix, config = T::linear_config(INTERMEDIATE, HIDDEN))]
    down_proj: Linear<T>,
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
    fn load<T: Activation>() -> Result<Self> {
        let dtype = T::KERNEL_DTYPE;
        Ok(Self {
            silu_mul: load_kernel(FUSE_SILU_MUL, || silu_mul_program(2, &[dtype; 2], &[dtype]))?,
            qk_norm_rope: load_kernel(FUSE_QK_NORM_ROPE, || {
                qk_norm_rope_program(3, &[dtype; 5], &[dtype; 2])
            })?,
            residual_norm: load_kernel(FUSE_RESIDUAL_NORM, || {
                residual_norm_program(2, &[dtype; 3], &[dtype; 2])
            })?,
            final_norm: load_kernel(FUSE_FINAL_NORM, || {
                final_norm_program(2, &[dtype; 2], &[dtype])
            })?,
        })
    }

    fn silu_mul(&self) -> Result<&Kernel> {
        required_kernel(&self.silu_mul, "silu_mul")
    }

    fn qk_norm_rope(&self) -> Result<&Kernel> {
        required_kernel(&self.qk_norm_rope, "qk_norm_rope")
    }

    fn residual_norm(&self) -> Result<&Kernel> {
        required_kernel(&self.residual_norm, "residual_norm")
    }

    fn final_norm(&self) -> Result<&Kernel> {
        required_kernel(&self.final_norm, "final_norm")
    }
}

fn load_kernel(enabled: bool, build: impl FnOnce() -> Result<Kernel>) -> Result<Option<Kernel>> {
    enabled.then(build).transpose()
}

fn required_kernel<'a>(kernel: &'a Option<Kernel>, name: &str) -> Result<&'a Kernel> {
    kernel
        .as_ref()
        .ok_or_else(|| forja_sdk::Error::loading(format!("fused kernel `{name}` is unavailable")))
}

struct LayerResources<'a, T: Activation> {
    cache: &'a mut KvCache<T>,
    following_norm: Option<&'a RmsNorm<T>>,
}

struct LayerPosition<'a, T: Activation> {
    positions: &'a Tensor<u32>,
    activation_positions: Option<&'a Tensor<T>>,
    sequence: u32,
    start: &'a Dim,
    end: &'a Dim,
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
        } = position;
        let query_projection = self.self_attn.q_proj.forward(normalized_input)?.reshape(&[
            sequence,
            QUERY_HEADS,
            HEAD_DIM,
        ])?;
        let key_projection = self.self_attn.k_proj.forward(normalized_input)?.reshape(&[
            sequence,
            KEY_VALUE_HEADS,
            HEAD_DIM,
        ])?;
        let value_projection = self.self_attn.v_proj.forward(normalized_input)?;
        let (query, key) = if FUSE_QK_NORM_ROPE {
            let program_positions = activation_positions.ok_or_else(|| {
                forja_sdk::Error::loading("fused rotary positions are unavailable")
            })?;
            (
                apply_qk_norm_rope(
                    kernels.qk_norm_rope()?,
                    &query_projection,
                    &self.self_attn.q_norm,
                    program_positions,
                )?,
                apply_qk_norm_rope(
                    kernels.qk_norm_rope()?,
                    &key_projection,
                    &self.self_attn.k_norm,
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
        let query = query.permute(&[1, 0, 2])?;
        let key = key.permute(&[1, 0, 2])?;
        let value = value_projection
            .reshape(&[sequence, KEY_VALUE_HEADS, HEAD_DIM])?
            .permute(&[1, 0, 2])?;
        let (cached_key, cached_value) = cache.append(&key, &value, start, sequence, end)?;
        let attended = sdpa(
            &query,
            &cached_key,
            &cached_value,
            ATTENTION_SCALE,
            true,
            start,
        )?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(&[sequence, QUERY_HEADS * HEAD_DIM])?;
        let attention = self.self_attn.o_proj.forward(&attended)?;
        let (hidden, normalized) = if FUSE_RESIDUAL_NORM {
            run_residual_norm(
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
        let gate = self.mlp.gate_proj.forward(&normalized)?;
        let up = self.mlp.up_proj.forward(&normalized)?;
        let activated = if FUSE_SILU_MUL {
            run_silu_mul(kernels.silu_mul()?, &gate, &up)?
        } else {
            gate.silu_mul(&up)?
        };
        let projected = self.mlp.down_proj.forward(&activated)?;
        if FUSE_RESIDUAL_NORM && let Some(norm) = following_norm {
            let (residual, normalized) =
                run_residual_norm(kernels.residual_norm()?, &hidden, &projected, norm.weight())?;
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

#[forja_sdk::kernel(map)]
fn silu_mul(gate: forja_sdk::kernel::Elem, up: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    gate * gate.sigmoid() * up
}

fn apply_qk_norm_rope<T: Activation>(
    kernel: &Kernel,
    input: &Tensor<T>,
    norm: &RmsNorm<T>,
    positions: &Tensor<T>,
) -> Result<Tensor<T>> {
    let [sequence, heads, head_dim]: [u32; 3] = input
        .shape()
        .try_into()
        .map_err(|_| forja_sdk::Error::loading("QK projection must have rank three"))?;
    let half = head_dim / 2;
    let shape = [sequence, heads, half];
    let lo = input.narrow(2, 0, half)?;
    let hi = input.narrow(2, half, half)?;
    let weight_lo = norm.weight().narrow(0, 0, half)?.broadcast_as(&shape)?;
    let weight_hi = norm.weight().narrow(0, half, half)?.broadcast_as(&shape)?;
    let positions = positions.reshape(&[sequence, 1, 1])?.broadcast_as(&shape)?;
    let output = Tensor::<T>::zeros(input.shape())?;
    let output_lo = output.narrow(2, 0, half)?;
    let output_hi = output.narrow(2, half, half)?;
    let inputs = [
        TensorRef::new(&lo)?,
        TensorRef::new(&hi)?,
        TensorRef::new(&weight_lo)?,
        TensorRef::new(&weight_hi)?,
        TensorRef::new(&positions)?,
    ];
    let outputs = [TensorRef::new(&output_lo)?, TensorRef::new(&output_hi)?];
    forja_sdk::kernel::run_into(kernel, &inputs, &outputs)?;
    Ok(output)
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

fn run_residual_norm<T: Activation>(
    kernel: &Kernel,
    residual: &Tensor<T>,
    update: &Tensor<T>,
    weight: &Tensor<T>,
) -> Result<(Tensor<T>, Tensor<T>)> {
    let weight = weight.broadcast_as(residual.shape())?;
    let inputs = [
        TensorRef::new(residual)?,
        TensorRef::new(update)?,
        TensorRef::new(&weight)?,
    ];
    let [value, normalized] = forja_sdk::kernel::run::<T, 2>(kernel, &inputs)?;
    Ok((value, normalized))
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

#[forja_sdk::kernel(row)]
fn qk_norm_rope(
    lo: forja_sdk::kernel::Row,
    hi: forja_sdk::kernel::Row,
    weight_lo: forja_sdk::kernel::Row,
    weight_hi: forja_sdk::kernel::Row,
    positions: forja_sdk::kernel::Row,
) -> (forja_sdk::kernel::Row, forja_sdk::kernel::Row) {
    let square_sum = (lo * lo + hi * hi).row_sum();
    let inverse_rms = (square_sum / 128.0 + RMS_EPSILON).rsqrt();
    let normalized_lo = lo * inverse_rms * weight_lo;
    let normalized_hi = hi * inverse_rms * weight_hi;
    let exponent = forja_sdk::kernel::index(-1) as f32 * (-2.0 / 128.0);
    let angle = positions * ROPE_THETA.powf(exponent);
    let cosine = angle.cos();
    let sine = angle.sin();
    (
        normalized_lo * cosine - normalized_hi * sine,
        normalized_hi * cosine + normalized_lo * sine,
    )
}

#[forja_sdk::kernel(row)]
fn residual_norm(
    residual: forja_sdk::kernel::Row,
    update: forja_sdk::kernel::Row,
    weight: forja_sdk::kernel::Row,
) -> (forja_sdk::kernel::Row, forja_sdk::kernel::Row) {
    let value = residual + update;
    let inverse_rms = (value * value).row_mean() + RMS_EPSILON;
    (value, value * inverse_rms.rsqrt() * weight)
}

#[forja_sdk::kernel(row)]
fn final_norm(
    input: forja_sdk::kernel::Row,
    weight: forja_sdk::kernel::Row,
) -> forja_sdk::kernel::Row {
    input * ((input * input).row_mean() + RMS_EPSILON).rsqrt() * weight
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
}

impl<T: Activation> Qwen3<T> {
    fn load_from_weights(weights: &Weights<'_>) -> Result<Self> {
        let weights = QwenWeights::<_>::load(weights, &Config)?;
        let caches = (0..LAYERS)
            .map(|_| KvCache::new(KEY_VALUE_HEADS, MAX_CONTEXT, HEAD_DIM, T::ZERO))
            .collect::<Result<Vec<_>>>()?;
        let kernels = FusedKernels::load::<T>()?;
        let positions = Tensor::constant(&(0..MAX_CONTEXT).collect::<Vec<_>>(), &[MAX_CONTEXT])?;
        let activation_positions = FUSE_QK_NORM_ROPE
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
        }
    }

    fn load(weights: &Weights<'_>) -> Result<Self> {
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
            });
        }
        self.forward(
            &input.tokens,
            sequence,
            input.start_pos.into(),
            end_pos.into(),
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
                if REPLAY_DECODE && sequence == 1 {
                    self.decode_selected(Some(&tokens), input.start_pos, selection)
                } else {
                    let logits = self
                        .forward(&tokens, sequence, input.start_pos.into(), end.into(), false)?
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
                let logits = self
                    .forward(&token, 1, input.start_pos.into(), end.into(), false)?
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
        start: Dim,
        end: Dim,
        taps_enabled: bool,
    ) -> Result<StepOutput> {
        let last = sequence
            .checked_sub(1)
            .ok_or_else(|| forja_sdk::Error::loading("tokens cannot be empty"))?;
        let positions = self.positions.narrow(0, &start, sequence)?;
        let program_positions = self
            .activation_positions
            .as_ref()
            .map(|positions| positions.narrow(0, &start, sequence))
            .transpose()?;
        let mut hidden = self.weights.model.embed_tokens.forward(tokens)?;
        let mut normalized = self.weights.model.layers[0]
            .input_layernorm
            .forward(&hidden)?;
        let mut taps = Vec::with_capacity(if taps_enabled { LAYERS } else { 0 });
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
                    start: &start,
                    end: &end,
                },
                LayerResources {
                    cache: &mut self.caches[index],
                    following_norm,
                },
            )?;
            hidden = next_hidden;
            if taps_enabled && index + 1 < LAYERS {
                taps.push(Self::output(hidden.contiguous()?)?);
            }
            if let Some(value) = next_normalized {
                normalized = value;
            }
        }
        hidden = if FUSE_FINAL_NORM {
            run_final_norm(
                self.kernels.final_norm()?,
                &hidden,
                self.weights.model.norm.weight(),
            )?
        } else {
            self.weights.model.norm.forward(&hidden)?
        };
        if taps_enabled {
            taps.push(Self::output(hidden.contiguous()?)?);
        }
        let logits = Self::output(
            self.weights
                .model
                .embed_tokens
                .project(&hidden)?
                .narrow(0, last, 1)?
                .reshape(&[VOCAB])?
                .contiguous()?,
        )?;
        Ok(StepOutput { logits, taps })
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
        let token = self.decode.token()?;
        let mut feedback = self.decode.token()?;
        let output_tokens = self.decode.output_tokens()?;
        let sampling = self.decode.sampling()?;
        forja_sdk::capture(&[&position], || {
            let logits = self
                .forward(&token, 1, start.clone().into(), end, false)?
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
