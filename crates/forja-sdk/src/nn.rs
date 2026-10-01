//! Small neural-network modules built from tensor operations.

use crate::{Element, Error, FloatElement, Load, Result, Tensor, Weights, bf16};

/// Applies softmax and deterministic top-k selection to router logits.
///
/// When `normalize` is true, the selected probabilities are renormalized to sum to one per row.
///
/// # Errors
///
/// Returns an error for an invalid top-k count, incompatible tensor shape, or refused dispatch.
pub fn moe_router<T: FloatElement>(
    logits: &Tensor<T>,
    k: u32,
    normalize: bool,
) -> Result<(Tensor<T>, Tensor<u32>)> {
    if normalize {
        logits.top_k_with_normalization(k, true)
    } else {
        logits.softmax_last_dim()?.top_k(k)
    }
}

/// Combines routed expert outputs with their per-route weights.
///
/// Expert outputs are `[rows, routes, hidden]` and weights are `[rows, routes]`.
///
/// # Errors
///
/// Returns an error for incompatible shapes or a refused view or matrix multiplication.
pub fn moe_combine<T: FloatElement>(experts: &Tensor<T>, weights: &Tensor<T>) -> Result<Tensor<T>> {
    let [rows, routes, hidden]: [u32; 3] = experts
        .shape()
        .try_into()
        .map_err(|_| Error::new("expert outputs must have rank three"))?;
    if weights.shape() != [rows, routes] {
        return Err(Error::new("expert weights do not match routed outputs"));
    }
    experts
        .permute(&[0, 2, 1])?
        .matmul(&weights.reshape(&[rows, routes, 1])?)?
        .reshape(&[rows, hidden])
}

/// Loads separate expert projections into `[experts, input, output]` storage.
///
/// Each source tensor is named `{expert}.{projection}.weight` and stored as `[output, input]`.
///
/// # Errors
///
/// Returns an error for invalid dimensions, missing weights, or a refused allocation or copy.
pub fn stack_expert_weights<T: WeightElement>(
    weights: &Weights<'_>,
    experts: u32,
    input: u32,
    output: u32,
    projection: &str,
) -> Result<Tensor<T>> {
    if experts == 0 || input == 0 || output == 0 {
        return Err(Error::loading(
            "expert projection dimensions must be nonzero",
        ));
    }
    let stacked = Tensor::zeros(&[experts, input, output])?;
    for expert in 0..experts {
        let source = weights
            .scoped(format!("{expert}.{projection}"))
            .tensor::<T>("weight", &[output, input])?
            .t()?;
        let mut destination = stacked.narrow(0, expert, 1)?.reshape(&[input, output])?;
        source.copy_into(&mut destination)?;
    }
    Ok(stacked)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WeightStorage {
    Activation,
    Bf16,
}

#[doc(hidden)]
pub trait WeightElement: Element {
    fn load_weight(
        weights: &Weights<'_>,
        name: &str,
        shape: &[u32],
        promote_bf16: bool,
    ) -> Result<Tensor<Self>>;
}

macro_rules! exact_weight_element {
    ($type:ty) => {
        impl WeightElement for $type {
            fn load_weight(
                weights: &Weights<'_>,
                name: &str,
                shape: &[u32],
                promote_bf16: bool,
            ) -> Result<Tensor<Self>> {
                if promote_bf16 {
                    Err(Error::loading(
                        "bf16 weight promotion requires f32 activations",
                    ))
                } else {
                    weights.tensor(name, shape)
                }
            }
        }
    };
}

exact_weight_element!(bf16);
exact_weight_element!(crate::f16);
exact_weight_element!(u32);
exact_weight_element!(i32);

impl WeightElement for f32 {
    fn load_weight(
        weights: &Weights<'_>,
        name: &str,
        shape: &[u32],
        promote_bf16: bool,
    ) -> Result<Tensor<Self>> {
        if promote_bf16 {
            weights.tensor::<bf16>(name, shape)?.to_dtype()
        } else {
            weights.tensor(name, shape)
        }
    }
}

/// Structured tensor operations.
pub mod ops;

/// Reusable transformer blocks.
pub mod blocks;

/// A linear projection with an `[output, input]` weight tensor.
pub struct Linear<T: Element> {
    weight: Tensor<T>,
}

/// Dimensions used to load a linear projection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LinearConfig {
    input: u32,
    output: u32,
    storage: WeightStorage,
}

impl LinearConfig {
    /// Describes an `input` to `output` projection.
    #[must_use]
    pub const fn new(input: u32, output: u32) -> Self {
        Self {
            input,
            output,
            storage: WeightStorage::Activation,
        }
    }

    /// Describes a projection stored as bf16 and promoted to f32 when loaded.
    #[must_use]
    pub const fn promoted_bf16(input: u32, output: u32) -> Self {
        Self {
            input,
            output,
            storage: WeightStorage::Bf16,
        }
    }
}

impl<T: WeightElement> Load<LinearConfig> for Linear<T> {
    fn load(weights: &Weights<'_>, config: &LinearConfig) -> Result<Self> {
        Ok(Self::new(T::load_weight(
            weights,
            "weight",
            &[config.output, config.input],
            config.storage == WeightStorage::Bf16,
        )?))
    }
}

impl<T: Element> Linear<T> {
    /// Creates a linear projection from its weight tensor.
    #[must_use]
    pub const fn new(weight: Tensor<T>) -> Self {
        Self { weight }
    }

    /// Applies the projection over the last input dimension.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid weight or input shape or a refused dispatch.
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        input.matmul(&self.weight.t()?)
    }
}

/// Root-mean-square normalization with a learned weight.
pub struct RmsNorm<T: Element> {
    weight: Tensor<T>,
    eps: f32,
}

/// Dimensions and epsilon used to load RMS normalization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RmsNormConfig {
    hidden: u32,
    eps: f32,
    storage: WeightStorage,
}

impl RmsNormConfig {
    /// Describes normalization over `hidden` values.
    #[must_use]
    pub const fn new(hidden: u32, eps: f32) -> Self {
        Self {
            hidden,
            eps,
            storage: WeightStorage::Activation,
        }
    }

    /// Describes bf16 normalization weights promoted to f32 when loaded.
    #[must_use]
    pub const fn promoted_bf16(hidden: u32, eps: f32) -> Self {
        Self {
            hidden,
            eps,
            storage: WeightStorage::Bf16,
        }
    }
}

impl<T: WeightElement> Load<RmsNormConfig> for RmsNorm<T> {
    fn load(weights: &Weights<'_>, config: &RmsNormConfig) -> Result<Self> {
        Ok(Self::new(
            T::load_weight(
                weights,
                "weight",
                &[config.hidden],
                config.storage == WeightStorage::Bf16,
            )?,
            config.eps,
        ))
    }
}

impl<T: Element> RmsNorm<T> {
    /// Creates a normalization module.
    #[must_use]
    pub const fn new(weight: Tensor<T>, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// Returns the learned scale tensor.
    #[must_use]
    pub const fn weight(&self) -> &Tensor<T> {
        &self.weight
    }

    /// Normalizes the input's last dimension.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, epsilon, types, or dispatches.
    pub fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        input.rms_norm(&self.weight, self.eps)
    }
}

/// A token embedding table.
pub struct Embedding<T: Element> {
    weight: Tensor<T>,
}

/// Dimensions used to load an embedding table.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmbeddingConfig {
    vocab: u32,
    hidden: u32,
    storage: WeightStorage,
}

impl EmbeddingConfig {
    /// Describes a `vocab` by `hidden` embedding table.
    #[must_use]
    pub const fn new(vocab: u32, hidden: u32) -> Self {
        Self {
            vocab,
            hidden,
            storage: WeightStorage::Activation,
        }
    }

    /// Describes a bf16 embedding table promoted to f32 when loaded.
    #[must_use]
    pub const fn promoted_bf16(vocab: u32, hidden: u32) -> Self {
        Self {
            vocab,
            hidden,
            storage: WeightStorage::Bf16,
        }
    }
}

impl<T: WeightElement> Load<EmbeddingConfig> for Embedding<T> {
    fn load(weights: &Weights<'_>, config: &EmbeddingConfig) -> Result<Self> {
        Ok(Self::new(T::load_weight(
            weights,
            "weight",
            &[config.vocab, config.hidden],
            config.storage == WeightStorage::Bf16,
        )?))
    }
}

impl<T: Element> Embedding<T> {
    /// Creates an embedding module from a `[vocabulary, hidden]` table.
    #[must_use]
    pub const fn new(weight: Tensor<T>) -> Self {
        Self { weight }
    }

    /// Looks up one row per token id.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid ids, shapes, types, or dispatches.
    pub fn forward(&self, ids: &Tensor<u32>) -> Result<Tensor<T>> {
        self.weight.embedding(ids)
    }

    /// Projects hidden states through the transposed embedding table.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid weight or input shape or a refused dispatch.
    pub fn project(&self, input: &Tensor<T>) -> Result<Tensor<T>> {
        input.matmul(&self.weight.t()?)
    }
}
