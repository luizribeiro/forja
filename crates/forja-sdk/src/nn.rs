//! Small neural-network modules built from tensor operations.

use crate::{Element, Error, Load, Result, Tensor, Weights, bf16};

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
