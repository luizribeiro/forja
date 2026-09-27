//! Small neural-network modules built from tensor operations.

use crate::{Element, Result, Tensor};

/// Structured tensor operations.
pub mod ops;

/// A linear projection with an `[output, input]` weight tensor.
pub struct Linear<T: Element> {
    weight: Tensor<T>,
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
