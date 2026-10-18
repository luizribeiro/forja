//! Qwen3-0.6B inference engine.

mod variants;

#[cfg(target_family = "wasm")]
use forja_sdk::Param;
#[cfg(target_family = "wasm")]
use forja_sdk::nn::blocks::{
    ChunkedPrefill, DEFAULT_PREFILL_CHUNK, DecodeSelection, DecodeState, PrefillExecution,
};
use forja_sdk::{
    DType, Dim, Engine, EngineInfo, FloatElement, Load, Result, StepInput, StepOutput, Tensor,
    VariantSelection, Weights, bf16, export_engine,
    kernel::Kernel,
    nn::{
        Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig, WeightElement,
        blocks::{
            KvCache, Taps, add, add_kernel, cached_attention_with, qk_norm_rope,
            qk_norm_rope_kernel, residual_norm, residual_norm_kernel, rms_norm, rms_norm_kernel,
            silu_mul, silu_mul_kernel,
        },
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

fn linear<T: Activation>(
    variants: &variants::Variants,
    projection: &Linear<T>,
    input: &Tensor<T>,
    rows: u32,
    inner: u32,
    columns: u32,
) -> Result<Tensor<T>> {
    let selection = variants.matmul(T::KERNEL_DTYPE, rows, inner, columns)?;
    projection.forward_with(input, &selection)
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
    fn project(
        &self,
        variants: &variants::Variants,
        input: &Tensor<T>,
        sequence: u32,
    ) -> Result<[Tensor<T>; 3]> {
        let query = linear(
            variants,
            &self.q_proj,
            input,
            sequence,
            HIDDEN,
            QUERY_HEADS * HEAD_DIM,
        )?
        .reshape(&[sequence, QUERY_HEADS, HEAD_DIM])?;
        let key = linear(
            variants,
            &self.k_proj,
            input,
            sequence,
            HIDDEN,
            KEY_VALUE_HEADS * HEAD_DIM,
        )?
        .reshape(&[sequence, KEY_VALUE_HEADS, HEAD_DIM])?;
        let value = linear(
            variants,
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
        kernels: &GuestKernels,
        variants: &variants::Variants,
        input: &Tensor<T>,
        sequence: u32,
    ) -> Result<Tensor<T>> {
        let gate = linear(
            variants,
            &self.gate_proj,
            input,
            sequence,
            HIDDEN,
            INTERMEDIATE,
        )?;
        let up = linear(
            variants,
            &self.up_proj,
            input,
            sequence,
            HIDDEN,
            INTERMEDIATE,
        )?;
        let activated = silu_mul(&kernels.silu_mul, &gate, &up)?;
        linear(
            variants,
            &self.down_proj,
            &activated,
            sequence,
            INTERMEDIATE,
            HIDDEN,
        )
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

struct GuestKernels {
    add: Kernel,
    silu_mul: Kernel,
    qk_norm_rope: Kernel,
    residual_norm: Kernel,
    rms_norm: Kernel,
}

impl GuestKernels {
    fn load<T: Activation>() -> Result<Self> {
        let dtype = T::KERNEL_DTYPE;
        Ok(Self {
            add: add_kernel(dtype)?,
            silu_mul: silu_mul_kernel(dtype)?,
            qk_norm_rope: qk_norm_rope_kernel(dtype, HEAD_DIM, RMS_EPSILON, ROPE_THETA)?,
            residual_norm: residual_norm_kernel(dtype, RMS_EPSILON)?,
            rms_norm: rms_norm_kernel(dtype, RMS_EPSILON)?,
        })
    }
}

struct LayerResources<'a, T: Activation> {
    cache: &'a mut KvCache<T>,
    following_norm: Option<&'a RmsNorm<T>>,
}

#[derive(Clone, Copy)]
struct LayerPosition<'a, T: Activation> {
    positions: &'a Tensor<T>,
    sequence: u32,
    start: &'a Dim,
    end: &'a Dim,
    variants: &'a variants::Variants,
    attention: &'a VariantSelection,
}

#[cfg(target_family = "wasm")]
#[derive(Clone, Copy)]
struct PrefillVariant<'a> {
    start: u32,
    parameter: Option<&'a Param>,
}

impl<T: Activation> DecoderLayer<T> {
    fn forward(
        &self,
        kernels: &GuestKernels,
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
            sequence,
            start,
            end,
            variants,
            attention,
        } = position;
        let [query_projection, key_projection, value_projection] =
            self.self_attn
                .project(variants, normalized_input, sequence)?;
        let query = qk_norm_rope(
            &kernels.qk_norm_rope,
            &query_projection,
            self.self_attn.q_norm.weight(),
            positions,
        )?;
        let key = qk_norm_rope(
            &kernels.qk_norm_rope,
            &key_projection,
            self.self_attn.k_norm.weight(),
            positions,
        )?;
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
            variants,
            &self.self_attn.o_proj,
            &attended,
            sequence,
            QUERY_HEADS * HEAD_DIM,
            HIDDEN,
        )?;
        let (hidden, normalized) = residual_norm(
            &kernels.residual_norm,
            input,
            &attention,
            self.post_attention_layernorm.weight(),
        )?;
        let projected = self.mlp.forward(kernels, variants, &normalized, sequence)?;
        if let Some(norm) = following_norm {
            let (residual, normalized) =
                residual_norm(&kernels.residual_norm, &hidden, &projected, norm.weight())?;
            Ok((residual, Some(normalized)))
        } else {
            Ok((add(&kernels.add, &hidden, &projected)?, None))
        }
    }
}

/// Qwen3-0.6B with a fixed 4096-token KV cache.
pub struct Qwen3<T: Activation = f32> {
    weights: QwenWeights<T>,
    variants: variants::Variants,
    caches: Vec<KvCache<T>>,
    kernels: GuestKernels,
    positions: Tensor<T>,
    #[cfg(target_family = "wasm")]
    replay: bool,
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
        let variants = variants::Variants::new(config, T::KERNEL_DTYPE)?;
        let weights = QwenWeights::<_>::load(weights, &Config)?;
        let caches = (0..LAYERS)
            .map(|_| KvCache::new(KEY_VALUE_HEADS, MAX_CONTEXT, HEAD_DIM, T::ZERO))
            .collect::<Result<Vec<_>>>()?;
        let kernels = GuestKernels::load::<T>()?;
        let positions = activation_position_table::<T>()?;
        Ok(Self {
            weights,
            variants,
            caches,
            kernels,
            positions,
            #[cfg(target_family = "wasm")]
            replay: config.replay,
            #[cfg(target_family = "wasm")]
            decode: DecodeState::new(MAX_CONTEXT, VOCAB)?,
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
        let hidden = self.weights.model.embed_tokens.forward(tokens)?;
        let normalized = rms_norm(
            &self.kernels.rms_norm,
            &hidden,
            self.weights.model.layers[0].input_layernorm.weight(),
        )?;
        let start = 0.into();
        let end = sequence.into();
        let attention = self.variants.attention(sequence, 0, None)?;
        self.weights.model.layers[0]
            .forward(
                &self.kernels,
                &hidden,
                &normalized,
                LayerPosition {
                    positions: &positions,
                    sequence,
                    start: &start,
                    end: &end,
                    variants: &self.variants,
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
        let end = end_pos.into();
        let attention = self.variants.attention(sequence, input.start_pos, None)?;
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
                        .forward(&tokens, sequence, &start, &end_dim, &attention, false)?
                        .logits;
                    self.select_token(logits, end - 1)
                }
            }
            None if input.start_pos < MAX_CONTEXT && self.replay => {
                self.decode_selected(None, input.start_pos, selection)
            }
            None if input.start_pos < MAX_CONTEXT => {
                let token = self.decode.token()?;
                let end = input.start_pos + 1;
                let start_dim = input.start_pos.into();
                let end_dim = end.into();
                let attention = self.variants.attention(1, input.start_pos, None)?;
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
        attention: &VariantSelection,
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
        attention: &VariantSelection,
        taps_enabled: bool,
    ) -> Result<(Tensor<T>, Vec<Tensor<f32>>)> {
        let positions = self.positions.narrow(0, start, sequence)?;
        let mut hidden = self.weights.model.embed_tokens.forward(tokens)?;
        let mut normalized = rms_norm(
            &self.kernels.rms_norm,
            &hidden,
            self.weights.model.layers[0].input_layernorm.weight(),
        )?;
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
                    sequence,
                    start,
                    end,
                    variants: &self.variants,
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
        hidden = rms_norm(
            &self.kernels.rms_norm,
            &hidden,
            self.weights.model.norm.weight(),
        )?;
        if taps.enabled() {
            taps.push(Self::output(hidden.contiguous()?)?);
        }
        let selection = self
            .variants
            .matmul(T::KERNEL_DTYPE, sequence, HIDDEN, VOCAB)?;
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
        let attention = self
            .variants
            .attention(sequence, variant.start, variant.parameter)?;
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
        let sample_variant = self.variants.sample()?;
        let attention = self.variants.attention(1, start_pos, Some(&position))?;
        forja_sdk::capture(&[&position], || {
            let logits = self
                .forward(&token, 1, &start_dim, &end, &attention, false)?
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
        let sample_variant = self.variants.sample()?;
        let selected = logits.reshape(&[1, VOCAB])?.sample_with(
            &self.decode.sampling()?,
            slot,
            &sample_variant,
        )?;
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
