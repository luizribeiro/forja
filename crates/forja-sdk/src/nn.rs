//! Small neural-network modules built from tensor operations.

use crate::{Element, Load, Result, Tensor, Weights};

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
}

impl LinearConfig {
    /// Describes an `input` to `output` projection.
    #[must_use]
    pub const fn new(input: u32, output: u32) -> Self {
        Self { input, output }
    }
}

impl<T: Element> Load<LinearConfig> for Linear<T> {
    fn load(weights: &Weights<'_>, config: &LinearConfig) -> Result<Self> {
        Ok(Self::new(
            weights.tensor("weight", &[config.output, config.input])?,
        ))
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
}

impl RmsNormConfig {
    /// Describes normalization over `hidden` values.
    #[must_use]
    pub const fn new(hidden: u32, eps: f32) -> Self {
        Self { hidden, eps }
    }
}

impl<T: Element> Load<RmsNormConfig> for RmsNorm<T> {
    fn load(weights: &Weights<'_>, config: &RmsNormConfig) -> Result<Self> {
        Ok(Self::new(
            weights.tensor("weight", &[config.hidden])?,
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
}

impl EmbeddingConfig {
    /// Describes a `vocab` by `hidden` embedding table.
    #[must_use]
    pub const fn new(vocab: u32, hidden: u32) -> Self {
        Self { vocab, hidden }
    }
}

impl<T: Element> Load<EmbeddingConfig> for Embedding<T> {
    fn load(weights: &Weights<'_>, config: &EmbeddingConfig) -> Result<Self> {
        Ok(Self::new(
            weights.tensor("weight", &[config.vocab, config.hidden])?,
        ))
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
}
