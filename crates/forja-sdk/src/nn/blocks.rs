//! Reusable transformer blocks.

use crate::{Dim, Element, Error, Result, Tensor};
#[cfg(target_family = "wasm")]
use crate::{Graph, SamplingParams};

use super::ops::sdpa;

/// Optional layer outputs collected for engine verification.
pub struct Taps {
    enabled: bool,
    values: Vec<Tensor<f32>>,
}

impl Taps {
    /// Creates a collector with capacity for every model layer when enabled.
    #[must_use]
    pub fn new(enabled: bool, layers: usize) -> Self {
        Self {
            enabled,
            values: Vec::with_capacity(if enabled { layers } else { 0 }),
        }
    }

    /// Reports whether layer outputs should be materialized.
    #[must_use]
    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Records one contiguous f32 layer output.
    pub fn push(&mut self, value: Tensor<f32>) {
        self.values.push(value);
    }

    /// Returns the collected outputs in layer order.
    #[must_use]
    pub fn finish(self) -> Vec<Tensor<f32>> {
        self.values
    }
}

/// Fixed-capacity key/value cache with `[heads, context, width]` storage.
pub struct KvCache<T: Element> {
    key: Tensor<T>,
    value: Tensor<T>,
}

/// Captured decode path selected for a request.
#[cfg(target_family = "wasm")]
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum DecodeSelection {
    /// Return logits without selecting a token.
    None,
    /// Select the greatest logit.
    Greedy,
    /// Sample from the configured distribution.
    Sampled,
}

/// Shared token buffers, sampling state, and captured decode graphs.
#[cfg(target_family = "wasm")]
pub struct DecodeState {
    token: Tensor<u32>,
    output_tokens: Tensor<u32>,
    sampling: Tensor<u32>,
    sampling_params: Option<SamplingParams>,
    graphs: [Option<Graph<Tensor<f32>>>; 3],
}

#[cfg(target_family = "wasm")]
impl DecodeState {
    /// Allocates decode state for a fixed context capacity.
    ///
    /// # Errors
    ///
    /// Returns an error when a buffer allocation is refused.
    pub fn new(max_context: u32) -> Result<Self> {
        Ok(Self {
            token: Tensor::zeros(&[1])?,
            output_tokens: Tensor::zeros(&[max_context])?,
            sampling: Tensor::zeros(&[5])?,
            sampling_params: None,
            graphs: [None, None, None],
        })
    }

    /// Updates sampling parameters and selects the matching decode path.
    ///
    /// # Errors
    ///
    /// Returns an error when the host refuses the sampling-state write.
    pub fn select(&mut self, params: SamplingParams) -> Result<DecodeSelection> {
        if self.sampling_params != Some(params) {
            let words = [
                params.temperature.to_bits(),
                params.top_k,
                params.top_p.to_bits(),
                params.seed as u32,
                (params.seed >> 32) as u32,
            ];
            self.sampling.write(&words)?;
            self.sampling_params = Some(params);
        }
        Ok(if params.temperature == 0.0 || params.top_k == 1 {
            DecodeSelection::Greedy
        } else {
            DecodeSelection::Sampled
        })
    }

    /// Replaces the current input token.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-scalar token tensor or a refused read or write.
    pub fn write_token(&self, token: &Tensor<u32>) -> Result<()> {
        self.token.write(&token.to_vec()?)
    }

    /// Returns an alias of the current token buffer.
    pub fn token(&self) -> Result<Tensor<u32>> {
        self.token.alias()
    }

    /// Returns an alias of the generated-token buffer.
    pub fn output_tokens(&self) -> Result<Tensor<u32>> {
        self.output_tokens.alias()
    }

    /// Returns an alias of the packed sampling parameters.
    pub fn sampling(&self) -> Result<Tensor<u32>> {
        self.sampling.alias()
    }

    /// Returns the graph captured for a decode path.
    #[must_use]
    pub fn graph(&self, selection: DecodeSelection) -> Option<&Graph<Tensor<f32>>> {
        self.graphs[selection as usize].as_ref()
    }

    /// Stores the graph captured for a decode path.
    pub fn set_graph(&mut self, selection: DecodeSelection, graph: Graph<Tensor<f32>>) {
        self.graphs[selection as usize] = Some(graph);
    }

    /// Copies a selected token into feedback and output storage.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid slot, shape, alias, or refused copy.
    pub fn store_selected(
        &self,
        selected: &Tensor<u32>,
        slot: impl Into<Dim>,
    ) -> Result<Tensor<u32>> {
        selected.copy_into(&mut self.token.alias()?)?;
        let mut output = self.output_tokens.narrow(0, slot, 1)?;
        selected.copy_into(&mut output)?;
        Ok(output)
    }

    /// Returns the selected token stored at `slot`.
    pub fn selected(&self, slot: impl Into<Dim>) -> Result<Tensor<u32>> {
        self.output_tokens.narrow(0, slot, 1)
    }
}

impl<T: Element> KvCache<T> {
    /// Allocates a zero-filled cache.
    ///
    /// # Errors
    ///
    /// Returns an error when the cache size overflows or allocation is refused.
    pub fn new(heads: u32, context: u32, width: u32, zero: T) -> Result<Self> {
        let count = u64::from(heads)
            .checked_mul(u64::from(context))
            .and_then(|count| count.checked_mul(u64::from(width)))
            .and_then(|count| usize::try_from(count).ok())
            .ok_or_else(|| Error::loading("KV cache size does not fit usize"))?;
        let zeros = vec![zero; count];
        let shape = [heads, context, width];
        Ok(Self {
            key: Tensor::from_slice(&zeros, &shape)?,
            value: Tensor::from_slice(&zeros, &shape)?,
        })
    }

    /// Appends projected keys and values and returns cache prefixes ending at `end`.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid ranges, shapes, aliasing, or refused copies.
    pub fn append(
        &mut self,
        key: &Tensor<T>,
        value: &Tensor<T>,
        start: &Dim,
        sequence: u32,
        end: &Dim,
    ) -> Result<(Tensor<T>, Tensor<T>)> {
        key.copy_into(&mut self.key.narrow(1, start, sequence)?)?;
        value.copy_into(&mut self.value.narrow(1, start, sequence)?)?;
        Ok((self.key.narrow(1, 0, end)?, self.value.narrow(1, 0, end)?))
    }
}

/// Applies cached causal attention to sequence-major query, key, and value tensors.
///
/// Inputs use `[sequence, heads, width]`; the result is `[sequence, query_heads * width]`.
///
/// # Errors
///
/// Returns an error for incompatible shapes, an invalid cache range, overflow, or refused work.
pub fn cached_attention<T: Element>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    cache: &mut KvCache<T>,
    scale: f32,
    start: &Dim,
    end: &Dim,
) -> Result<Tensor<T>> {
    let [sequence, query_heads, width]: [u32; 3] = query
        .shape()
        .try_into()
        .map_err(|_| Error::loading("attention query must have rank three"))?;
    let key_shape: [u32; 3] = key
        .shape()
        .try_into()
        .map_err(|_| Error::loading("attention key must have rank three"))?;
    if key_shape[0] != sequence || key_shape[2] != width || value.shape() != key.shape() {
        return Err(Error::loading(
            "attention key and value shapes are incompatible",
        ));
    }
    let query = query.permute(&[1, 0, 2])?;
    let key = key.permute(&[1, 0, 2])?;
    let value = value.permute(&[1, 0, 2])?;
    let (cached_key, cached_value) = cache.append(&key, &value, start, sequence, end)?;
    let hidden = query_heads
        .checked_mul(width)
        .ok_or_else(|| Error::loading("attention output width overflowed"))?;
    sdpa(&query, &cached_key, &cached_value, scale, true, start)?
        .permute(&[1, 0, 2])?
        .contiguous()?
        .reshape(&[sequence, hidden])
}
