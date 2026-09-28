//! Qwen3-0.6B inference engine.

#[cfg(any(
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

use forja_sdk::{
    DType, Engine, EngineInfo, Load, Result, StepInput, StepOutput, Tensor, Weights, bf16,
    export_engine,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, WeightElement,
        ops::sdpa,
    },
    program::{Kernel, Program, ReduceOp, Value},
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
const FUSE_RESIDUAL_NORM: bool = cfg!(feature = "all-fusions")
    || !cfg!(any(
        feature = "qk-norm-rope-only",
        feature = "silu-mul-only",
        feature = "final-norm-only"
    ));
const FUSE_QK_NORM_ROPE: bool = cfg!(any(feature = "qk-norm-rope-only", feature = "all-fusions"));
const FUSE_SILU_MUL: bool = cfg!(any(feature = "silu-mul-only", feature = "all-fusions"));
const FUSE_FINAL_NORM: bool = cfg!(any(feature = "final-norm-only", feature = "all-fusions"));

#[derive(Clone, Copy)]
struct Config;

struct Kernels {
    residual_norm: Kernel,
    qk_norm_rope: Kernel,
    silu_mul: Kernel,
    final_norm: Kernel,
}

impl Kernels {
    fn new<T: Activation>() -> Result<Self> {
        Ok(Self {
            residual_norm: prepare_kernel::<T>(&residual_norm_program(), 2, 3, 2)?,
            qk_norm_rope: prepare_kernel::<T>(&qk_norm_rope_program()?, 3, 5, 2)?,
            silu_mul: prepare_kernel::<T>(&silu_mul_program(), 2, 2, 1)?,
            final_norm: prepare_kernel::<T>(&final_norm_program(), 2, 2, 1)?,
        })
    }
}

/// Activation precision supported by the Qwen3 engine.
pub trait Activation: WeightElement {
    /// Additive identity used to initialize the KV cache.
    const ZERO: Self;

    /// Element type used by prepared activation kernels.
    const DTYPE: DType;

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
    const DTYPE: DType = DType::F32;

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
    const DTYPE: DType = DType::BF16;

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

struct LayerCache<T: Activation> {
    key: Tensor<T>,
    value: Tensor<T>,
}

struct LayerResources<'a, T: Activation> {
    cache: &'a mut LayerCache<T>,
    following_norm: Option<&'a RmsNorm<T>>,
    kernels: &'a Kernels,
}

impl<T: Activation> LayerCache<T> {
    fn new() -> Result<Self> {
        let shape = [KEY_VALUE_HEADS, MAX_CONTEXT, HEAD_DIM];
        let count = usize::try_from(
            u64::from(KEY_VALUE_HEADS) * u64::from(MAX_CONTEXT) * u64::from(HEAD_DIM),
        )
        .map_err(|_| forja_sdk::Error::loading("KV cache size does not fit usize"))?;
        let zeros = vec![T::ZERO; count];
        Ok(Self {
            key: Tensor::from_slice(&zeros, &shape)?,
            value: Tensor::from_slice(&zeros, &shape)?,
        })
    }
}

impl<T: Activation> DecoderLayer<T> {
    fn forward(
        &self,
        input: &Tensor<T>,
        normalized_input: &Tensor<T>,
        positions: (&Tensor<u32>, Option<&Tensor<T>>),
        positions_range: std::ops::Range<u32>,
        resources: LayerResources<'_, T>,
    ) -> Result<(Tensor<T>, Option<Tensor<T>>)> {
        let LayerResources {
            cache,
            following_norm,
            kernels,
        } = resources;
        let sequence = positions_range.end - positions_range.start;
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
            let program_positions = positions.1.ok_or_else(|| {
                forja_sdk::Error::loading("fused rotary positions are unavailable")
            })?;
            (
                qk_norm_rope(
                    &query_projection,
                    &self.self_attn.q_norm,
                    program_positions,
                    &kernels.qk_norm_rope,
                )?,
                qk_norm_rope(
                    &key_projection,
                    &self.self_attn.k_norm,
                    program_positions,
                    &kernels.qk_norm_rope,
                )?,
            )
        } else {
            (
                self.self_attn
                    .q_norm
                    .forward(&query_projection)?
                    .rope(positions.0, ROPE_THETA)?,
                self.self_attn
                    .k_norm
                    .forward(&key_projection)?
                    .rope(positions.0, ROPE_THETA)?,
            )
        };
        let query = query.permute(&[1, 0, 2])?;
        let key = key.permute(&[1, 0, 2])?;
        let value = value_projection
            .reshape(&[sequence, KEY_VALUE_HEADS, HEAD_DIM])?
            .permute(&[1, 0, 2])?;
        key.copy_into(&mut cache.key.narrow(1, positions_range.start, sequence)?)?;
        value.copy_into(&mut cache.value.narrow(1, positions_range.start, sequence)?)?;
        let attended = sdpa(
            &query,
            &cache.key.narrow(1, 0, positions_range.end)?,
            &cache.value.narrow(1, 0, positions_range.end)?,
            ATTENTION_SCALE,
            true,
            positions_range.start,
        )?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(&[sequence, QUERY_HEADS * HEAD_DIM])?;
        let attention = self.self_attn.o_proj.forward(&attended)?;
        let (hidden, normalized) = if FUSE_RESIDUAL_NORM {
            residual_norm(
                input,
                &attention,
                &self.post_attention_layernorm,
                &kernels.residual_norm,
            )?
        } else {
            let hidden = (input + &attention)?;
            let normalized = self.post_attention_layernorm.forward(&hidden)?;
            (hidden, normalized)
        };
        let gate = self.mlp.gate_proj.forward(&normalized)?;
        let up = self.mlp.up_proj.forward(&normalized)?;
        let activated = if FUSE_SILU_MUL {
            silu_mul(&gate, &up, &kernels.silu_mul)?
        } else {
            gate.silu_mul(&up)?
        };
        let projected = self.mlp.down_proj.forward(&activated)?;
        if FUSE_RESIDUAL_NORM && let Some(norm) = following_norm {
            let (residual, normalized) =
                residual_norm(&hidden, &projected, norm, &kernels.residual_norm)?;
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

fn silu_mul<T: Activation>(gate: &Tensor<T>, up: &Tensor<T>, kernel: &Kernel) -> Result<Tensor<T>> {
    let [output] = gate
        .run_kernel(kernel, &[up])?
        .try_into()
        .map_err(|_| forja_sdk::Error::loading("SiLU multiplication produced invalid outputs"))?;
    Ok(output)
}

fn silu_mul_program() -> Program {
    let program = Program::map();
    let gate_value = program.input(0);
    program.output(0, gate_value * gate_value.sigmoid() * program.input(1));
    program
}

fn qk_norm_rope<T: Activation>(
    input: &Tensor<T>,
    norm: &RmsNorm<T>,
    positions: &Tensor<T>,
    kernel: &Kernel,
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
    lo.run_kernel_into(
        kernel,
        &[&hi, &weight_lo, &weight_hi, &positions],
        &[&output_lo, &output_hi],
    )?;
    Ok(output)
}

fn qk_norm_rope_program() -> Result<Program> {
    let head_dimension = u16::try_from(HEAD_DIM)
        .map(f32::from)
        .map_err(|_| forja_sdk::Error::loading("head dimension exceeds the program range"))?;
    let program = Program::row();
    let lo_value = program.input(0);
    let hi_value = program.input(1);
    let square_sum = program.reduce(ReduceOp::Sum, lo_value * lo_value + hi_value * hi_value);
    let inverse_rms = (square_sum / head_dimension + RMS_EPSILON).rsqrt();
    let normalized_lo = lo_value * inverse_rms * program.input(2);
    let normalized_hi = hi_value * inverse_rms * program.input(3);
    let exponent = program.index(-1) * (-2.0 / head_dimension);
    let angle = program.input(4) * program.constant(ROPE_THETA).pow(exponent);
    let cosine = angle.cos();
    let sine = angle.sin();
    program.output(0, normalized_lo * cosine - normalized_hi * sine);
    program.output(1, normalized_hi * cosine + normalized_lo * sine);
    Ok(program)
}

fn residual_norm<T: Activation>(
    residual: &Tensor<T>,
    update: &Tensor<T>,
    norm: &RmsNorm<T>,
    kernel: &Kernel,
) -> Result<(Tensor<T>, Tensor<T>)> {
    let weight = norm.weight().broadcast_as(residual.shape())?;
    let outputs = residual.run_kernel(kernel, &[update, &weight])?;
    let [residual, normalized] = outputs
        .try_into()
        .map_err(|_| forja_sdk::Error::loading("residual norm produced invalid outputs"))?;
    Ok((residual, normalized))
}

fn residual_norm_program() -> Program {
    let program = Program::row();
    let value = program.input(0) + program.input(1);
    program.output(0, value);
    program.output(1, rms_normalize(&program, value, program.input(2)));
    program
}

fn final_norm<T: Activation>(
    input: &Tensor<T>,
    norm: &RmsNorm<T>,
    kernel: &Kernel,
) -> Result<Tensor<T>> {
    let weight = norm.weight().broadcast_as(input.shape())?;
    let [output] = input
        .run_kernel(kernel, &[&weight])?
        .try_into()
        .map_err(|_| forja_sdk::Error::loading("final norm produced invalid outputs"))?;
    Ok(output)
}

fn final_norm_program() -> Program {
    let program = Program::row();
    let value = program.input(0);
    program.output(0, rms_normalize(&program, value, program.input(1)));
    program
}

fn prepare_kernel<T: Activation>(
    program: &Program,
    rank: u8,
    inputs: usize,
    outputs: usize,
) -> Result<Kernel> {
    Kernel::new(
        program,
        rank,
        &vec![<T as Activation>::DTYPE; inputs],
        &vec![<T as Activation>::DTYPE; outputs],
    )
}

fn rms_normalize<'a>(program: &'a Program, value: Value<'a>, weight: Value<'a>) -> Value<'a> {
    let square_sum = program.reduce(ReduceOp::Sum, value * value);
    let inverse_rms = (square_sum / program.extent(-1) + RMS_EPSILON).rsqrt();
    value * inverse_rms * weight
}

/// Qwen3-0.6B with a fixed 4096-token KV cache.
pub struct Qwen3<T: Activation = f32> {
    weights: QwenWeights<T>,
    caches: Vec<LayerCache<T>>,
    kernels: Kernels,
}

impl<T: Activation> Qwen3<T> {
    fn load_from_weights(weights: &Weights<'_>) -> Result<Self> {
        let kernels = Kernels::new::<T>()?;
        let weights = QwenWeights::<_>::load(weights, &Config)?;
        let caches = (0..LAYERS)
            .map(|_| LayerCache::new())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            weights,
            caches,
            kernels,
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
        let positions = positions(0, sequence)?;
        let program_positions = if FUSE_QK_NORM_ROPE {
            Some(activation_positions::<T>(0, sequence)?)
        } else {
            None
        };
        let hidden = self.weights.model.embed_tokens.forward(tokens)?;
        let normalized = self.weights.model.layers[0]
            .input_layernorm
            .forward(&hidden)?;
        self.weights.model.layers[0]
            .forward(
                &hidden,
                &normalized,
                (&positions, program_positions.as_ref()),
                0..sequence,
                LayerResources {
                    cache: &mut self.caches[0],
                    following_norm: None,
                    kernels: &self.kernels,
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
        let last = sequence
            .checked_sub(1)
            .ok_or_else(|| forja_sdk::Error::loading("tokens cannot be empty"))?;
        let end_pos = input
            .start_pos
            .checked_add(sequence)
            .filter(|&end| end <= MAX_CONTEXT)
            .ok_or_else(|| forja_sdk::Error::loading("tokens exceed the 4096-token context"))?;
        let positions = positions(input.start_pos, end_pos)?;
        let program_positions = if FUSE_QK_NORM_ROPE {
            Some(activation_positions(input.start_pos, end_pos)?)
        } else {
            None
        };
        let mut hidden = self.weights.model.embed_tokens.forward(&input.tokens)?;
        let mut normalized = self.weights.model.layers[0]
            .input_layernorm
            .forward(&hidden)?;
        let mut taps = Vec::with_capacity(if input.taps { LAYERS } else { 0 });
        for index in 0..LAYERS {
            let layer = &self.weights.model.layers[index];
            let following_norm = self
                .weights
                .model
                .layers
                .get(index + 1)
                .map(|layer| &layer.input_layernorm);
            let (next_hidden, next_normalized) = layer.forward(
                &hidden,
                &normalized,
                (&positions, program_positions.as_ref()),
                input.start_pos..end_pos,
                LayerResources {
                    cache: &mut self.caches[index],
                    following_norm,
                    kernels: &self.kernels,
                },
            )?;
            hidden = next_hidden;
            if input.taps && index + 1 < LAYERS {
                taps.push(Self::output(hidden.contiguous()?)?);
            }
            if let Some(value) = next_normalized {
                normalized = value;
            }
        }
        hidden = if FUSE_FINAL_NORM {
            final_norm(&hidden, &self.weights.model.norm, &self.kernels.final_norm)?
        } else {
            self.weights.model.norm.forward(&hidden)?
        };
        if input.taps {
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
}

fn positions(start: u32, end: u32) -> Result<Tensor<u32>> {
    Tensor::from_slice(&(start..end).collect::<Vec<_>>(), &[end - start])
}

fn activation_positions<T: Activation>(start: u32, end: u32) -> Result<Tensor<T>> {
    let values = (start..end)
        .map(|position| {
            u16::try_from(position)
                .map(T::from_position)
                .map_err(|_| forja_sdk::Error::loading("position exceeds activation range"))
        })
        .collect::<Result<Vec<_>>>()?;
    Tensor::from_slice(&values, &[end - start])
}
