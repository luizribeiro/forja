//! Reusable transformer blocks.

#[cfg(target_family = "wasm")]
use crate::SamplingParams;
use crate::{
    DType, Dim, Element, Error, FloatElement, Graph, Param, Result, Slice, Tensor,
    kernel::{Kernel, TensorRef},
};

use super::ops::{sdpa, sdpa_into, sdpa_into_with};

/// Default number of tokens recorded in a full prefill graph.
pub const DEFAULT_PREFILL_CHUNK: u32 = 512;

const MIN_PREFILL_BUCKET: u32 = 16;

/// Builds the row program for RMS normalization.
///
/// # Errors
///
/// Returns an error when the kernel definition is refused.
pub fn rms_norm_kernel(dtype: DType, epsilon: f32) -> Result<Kernel> {
    rms_norm_rows_program(2, &[dtype; 2], &[dtype], epsilon)
}

/// Applies RMS normalization with a prebuilt row kernel.
///
/// # Errors
///
/// Returns an error for incompatible tensors or a refused kernel dispatch.
pub fn rms_norm<T: FloatElement>(
    kernel: &Kernel,
    input: &Tensor<T>,
    weight: &Tensor<T>,
) -> Result<Tensor<T>> {
    let weight = weight.broadcast_as(input.shape())?;
    let inputs = [TensorRef::new(input)?, TensorRef::new(&weight)?];
    let [output] = crate::kernel::run::<T, 1>(kernel, &inputs)?;
    Ok(output)
}

/// Builds the row program that combines a residual addition with RMS normalization.
///
/// # Errors
///
/// Returns an error when the kernel definition is refused.
pub fn residual_norm_kernel(dtype: DType, epsilon: f32) -> Result<Kernel> {
    residual_norm_rows_program(2, &[dtype; 3], &[dtype; 2], epsilon)
}

/// Adds an update to a residual and returns both the sum and its RMS-normalized value.
///
/// # Errors
///
/// Returns an error for incompatible tensors or a refused kernel dispatch.
pub fn residual_norm<T: FloatElement>(
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
    let [value, normalized] = crate::kernel::run::<T, 2>(kernel, &inputs)?;
    Ok((value, normalized))
}

/// Builds the row program that combines RMS normalization with rotary embedding.
///
/// # Errors
///
/// Returns an error when the kernel definition is refused.
#[allow(clippy::cast_precision_loss)]
pub fn qk_norm_rope_kernel(
    dtype: DType,
    head_dim: u32,
    epsilon: f32,
    theta: f32,
) -> Result<Kernel> {
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(Error::loading("QK head width must be positive and even"));
    }
    qk_norm_rope_program::qk_norm_rope_rows_program(
        3,
        &[dtype; 5],
        &[dtype; 2],
        head_dim as f32,
        epsilon,
        theta,
    )
}

/// Applies per-head RMS normalization and rotary embedding to a Q/K projection.
///
/// # Errors
///
/// Returns an error for an invalid projection shape or a refused view or kernel dispatch.
pub fn qk_norm_rope<T: FloatElement>(
    kernel: &Kernel,
    input: &Tensor<T>,
    weight: &Tensor<T>,
    positions: &Tensor<T>,
) -> Result<Tensor<T>> {
    let [sequence, heads, head_dim]: [u32; 3] = input
        .shape()
        .try_into()
        .map_err(|_| Error::loading("QK projection must have rank three"))?;
    if head_dim == 0 || !head_dim.is_multiple_of(2) {
        return Err(Error::loading("QK head width must be positive and even"));
    }
    let half = head_dim / 2;
    let shape = [sequence, heads, half];
    let lo = input.narrow(2, 0, half)?;
    let hi = input.narrow(2, half, half)?;
    let (weight_lo, weight_hi) = match weight.shape() {
        [width] if *width == head_dim => (
            weight.narrow(0, 0, half)?.broadcast_as(&shape)?,
            weight.narrow(0, half, half)?.broadcast_as(&shape)?,
        ),
        [weight_heads, width] if *weight_heads == heads && *width == head_dim => {
            let weight = weight.reshape(&[1, heads, head_dim])?;
            (
                weight.narrow(2, 0, half)?.broadcast_as(&shape)?,
                weight.narrow(2, half, half)?.broadcast_as(&shape)?,
            )
        }
        _ => return Err(Error::loading("QK norm weight shape is incompatible")),
    };
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
    crate::kernel::run_into(kernel, &inputs, &outputs)?;
    Ok(output)
}

/// Pairing convention for rotary position embeddings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RopeStyle {
    /// Pairs the lower and upper halves of the rotary prefix.
    HalfSplit,
    /// Pairs adjacent even and odd elements of the rotary prefix.
    Interleaved,
}

/// Guest rotary-embedding configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeConfig {
    /// Element pairing convention.
    pub style: RopeStyle,
    /// Rotated prefix width, or the full head width when omitted.
    pub rotary_dim: Option<u32>,
    /// Frequency base.
    pub theta: f32,
}

impl RopeConfig {
    /// Configures full-width rotary embeddings.
    #[must_use]
    pub const fn new(style: RopeStyle, theta: f32) -> Self {
        Self {
            style,
            rotary_dim: None,
            theta,
        }
    }

    /// Restricts rotation to a prefix of each head.
    #[must_use]
    pub const fn with_rotary_dim(mut self, rotary_dim: u32) -> Self {
        self.rotary_dim = Some(rotary_dim);
        self
    }

    fn validate(self, head_dim: u32) -> Result<u32> {
        let rotary_dim = self.rotary_dim.unwrap_or(head_dim);
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) || rotary_dim > head_dim {
            return Err(Error::loading(
                "rotary width must be positive, even, and within the head width",
            ));
        }
        if !self.theta.is_finite() || self.theta <= 0.0 {
            return Err(Error::loading("rotary theta must be finite and positive"));
        }
        Ok(rotary_dim)
    }
}

/// Builds a guest Map program for rotary position embeddings.
///
/// # Errors
///
/// Returns an error for invalid head or rotary widths, theta, or kernel definition.
#[allow(clippy::cast_precision_loss)]
pub fn rope_kernel(dtype: DType, head_dim: u32, config: RopeConfig) -> Result<Kernel> {
    let rotary_dim = config.validate(head_dim)?;
    rope_program::rope_pairs_program(3, &[dtype; 3], &[dtype; 2], rotary_dim as f32, config.theta)
}

/// Applies a guest rotary Map program to a sequence-major head tensor.
///
/// # Errors
///
/// Returns an error for incompatible shapes or configuration, or a refused view, copy, or kernel
/// dispatch.
pub fn rope<T: FloatElement>(
    kernel: &Kernel,
    input: &Tensor<T>,
    positions: &Tensor<T>,
    config: RopeConfig,
) -> Result<Tensor<T>> {
    let [sequence, heads, head_dim]: [u32; 3] = input
        .shape()
        .try_into()
        .map_err(|_| Error::loading("rotary input must have rank three"))?;
    if positions.shape() != [sequence] {
        return Err(Error::loading(
            "rotary positions must match the sequence width",
        ));
    }
    let rotary_dim = config.validate(head_dim)?;
    let mut output = Tensor::<T>::empty(input.shape().to_vec())?;
    if rotary_dim < head_dim {
        input.copy_into(&mut output)?;
    }
    let (input_first, input_second) = rotary_pair_views(input, config.style, rotary_dim)?;
    let pair_shape = [sequence, heads, rotary_dim / 2];
    let positions = positions
        .reshape(&[sequence, 1, 1])?
        .broadcast_as(&pair_shape)?;
    let (output_first, output_second, copy_outputs) = match config.style {
        RopeStyle::HalfSplit => {
            let (first, second) = rotary_pair_views(&output, config.style, rotary_dim)?;
            (first, second, false)
        }
        RopeStyle::Interleaved => (
            Tensor::<T>::empty(pair_shape.to_vec())?,
            Tensor::<T>::empty(pair_shape.to_vec())?,
            true,
        ),
    };
    crate::kernel::run_into(
        kernel,
        &[
            TensorRef::new(&input_first)?,
            TensorRef::new(&input_second)?,
            TensorRef::new(&positions)?,
        ],
        &[
            TensorRef::new(&output_first)?,
            TensorRef::new(&output_second)?,
        ],
    )?;
    if copy_outputs {
        let (mut first, mut second) = rotary_pair_views(&output, config.style, rotary_dim)?;
        output_first.copy_into(&mut first)?;
        output_second.copy_into(&mut second)?;
    }
    Ok(output)
}

fn rotary_pair_views<T: Element>(
    tensor: &Tensor<T>,
    style: RopeStyle,
    rotary_dim: u32,
) -> Result<(Tensor<T>, Tensor<T>)> {
    let [sequence, heads, _]: [u32; 3] = tensor
        .shape()
        .try_into()
        .map_err(|_| Error::loading("rotary tensor must have rank three"))?;
    let half = rotary_dim / 2;
    match style {
        RopeStyle::HalfSplit => Ok((tensor.narrow(2, 0, half)?, tensor.narrow(2, half, half)?)),
        RopeStyle::Interleaved => Ok((
            tensor.slice(&[
                Slice::new(0, sequence, 1)?,
                Slice::new(0, heads, 1)?,
                Slice::new(0, half, 2)?,
            ])?,
            tensor.slice(&[
                Slice::new(0, sequence, 1)?,
                Slice::new(0, heads, 1)?,
                Slice::new(1, half, 2)?,
            ])?,
        )),
    }
}

#[crate::kernel::kernel(row)]
fn residual_norm_rows(
    residual: crate::kernel::Row,
    update: crate::kernel::Row,
    weight: crate::kernel::Row,
    epsilon: f32,
) -> (crate::kernel::Row, crate::kernel::Row) {
    let value = residual + update;
    let inverse_rms = (value * value).row_mean() + epsilon;
    (value, value * inverse_rms.rsqrt() * weight)
}

#[crate::kernel::kernel(row)]
fn rms_norm_rows(
    input: crate::kernel::Row,
    weight: crate::kernel::Row,
    epsilon: f32,
) -> crate::kernel::Row {
    let inverse_rms = (input * input).row_mean() + epsilon;
    input * inverse_rms.rsqrt() * weight
}

#[allow(clippy::too_many_arguments)]
mod qk_norm_rope_program {
    #[crate::kernel::kernel(row)]
    pub(super) fn qk_norm_rope_rows(
        lo: crate::kernel::Row,
        hi: crate::kernel::Row,
        weight_lo: crate::kernel::Row,
        weight_hi: crate::kernel::Row,
        positions: crate::kernel::Row,
        head_dim: f32,
        epsilon: f32,
        theta: f32,
    ) -> (crate::kernel::Row, crate::kernel::Row) {
        let square_sum = (lo * lo + hi * hi).row_sum();
        let inverse_rms = (square_sum / head_dim + epsilon).rsqrt();
        let normalized_lo = lo * inverse_rms * weight_lo;
        let normalized_hi = hi * inverse_rms * weight_hi;
        let exponent = crate::kernel::index(-1) as f32 * (-2.0 / head_dim);
        let angle = positions * theta.powf(exponent);
        let cosine = angle.cos();
        let sine = angle.sin();
        (
            normalized_lo * cosine - normalized_hi * sine,
            normalized_hi * cosine + normalized_lo * sine,
        )
    }
}

mod rope_program {
    #[crate::kernel::kernel(map)]
    pub(super) fn rope_pairs(
        first: crate::kernel::Elem,
        second: crate::kernel::Elem,
        positions: crate::kernel::Elem,
        rotary_dim: f32,
        theta: f32,
    ) -> (crate::kernel::Elem, crate::kernel::Elem) {
        let exponent = crate::kernel::index(-1) as f32 * (-2.0 / rotary_dim);
        let angle = positions * theta.powf(exponent);
        let cosine = angle.cos();
        let sine = angle.sin();
        (
            first * cosine - second * sine,
            second * cosine + first * sine,
        )
    }
}

struct CapturedPrefill<T: Element> {
    graph: Graph<Tensor<T>>,
}

struct PrefillTail<T: Element> {
    size: u32,
    captured: Option<CapturedPrefill<T>>,
}

/// Lazily captured prefill graphs and their retained token buffer.
pub struct ChunkedPrefill<T: Element> {
    max_context: u32,
    chunk: u32,
    tokens: Tensor<u32>,
    full: Option<CapturedPrefill<T>>,
    tails: Vec<PrefillTail<T>>,
}

impl<T: Element> ChunkedPrefill<T> {
    /// Allocates a retained token buffer and one replay graph slot per chunk shape.
    ///
    /// # Errors
    ///
    /// Returns an error unless `chunk` is a power of two from 16 through 512, or
    /// when a retained token buffer cannot be allocated.
    pub fn new(max_context: u32, chunk: u32) -> Result<Self> {
        if !(MIN_PREFILL_BUCKET..=DEFAULT_PREFILL_CHUNK).contains(&chunk)
            || !chunk.is_power_of_two()
            || chunk > max_context
        {
            return Err(Error::loading(
                "prefill chunk must be a power of two from 16 through 512 within the context",
            ));
        }
        let mut tails = Vec::new();
        let mut size = MIN_PREFILL_BUCKET;
        while size < chunk {
            tails.push(PrefillTail {
                size,
                captured: None,
            });
            size = size
                .checked_mul(2)
                .ok_or_else(|| Error::loading("prefill bucket size overflowed"))?;
        }
        Ok(Self {
            max_context,
            chunk,
            tokens: Tensor::zeros(&[max_context])?,
            full: None,
            tails,
        })
    }

    /// Reports whether the padded token range stays inside the cache.
    #[must_use]
    pub fn supports(&self, start: u32, sequence: u32) -> bool {
        sequence != 0
            && padded_prefill_len(sequence, self.chunk)
                .and_then(|padded| start.checked_add(padded))
                .is_some_and(|end| end <= self.max_context)
    }

    /// Writes the prompt once and replays one graph for every chunk of the same shape.
    ///
    /// Capture is lazy: the first use records one graph for the entire bucket.
    /// Padding follows real tokens, so causal attention and per-token `MoE` routing
    /// cannot change real outputs. Padded cache entries are safe because the next
    /// prefill or decode writes its position before attending to it. `forward`
    /// must return the last real token's logits.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, unsupported cache padding, malformed
    /// logits, or a refused tensor, capture, or replay operation.
    pub fn replay(
        &mut self,
        tokens: &Tensor<u32>,
        start: u32,
        mut forward: impl FnMut(&Tensor<u32>, u32, &Dim, &Dim, &Dim, &Param) -> Result<Tensor<T>>,
    ) -> Result<Tensor<T>> {
        let [sequence] = tokens
            .shape()
            .try_into()
            .map_err(|_| Error::loading("prefill tokens must have rank one"))?;
        if sequence == 0 {
            return Err(Error::loading("prefill tokens cannot be empty"));
        }
        if !self.supports(start, sequence) {
            return Err(Error::loading("prefill bucket exceeds the context"));
        }
        let values = tokens.to_vec()?;
        let plan = prefill_plan(sequence, self.chunk)?;
        self.tokens
            .write(&staged_tokens(&values, start, self.max_context)?)?;
        let mut consumed = 0_u32;
        let mut position = start;
        let mut result = None;
        for (real, bucket_size) in plan {
            let captured = if bucket_size == self.chunk {
                &mut self.full
            } else {
                &mut self
                    .tails
                    .iter_mut()
                    .find(|tail| tail.size == bucket_size)
                    .ok_or_else(|| Error::loading("prefill tail was not allocated"))?
                    .captured
            };
            if captured.is_none() {
                let position_parameter = Param::new(0..=self.max_context - bucket_size)?;
                let last_parameter = Param::new(0..=bucket_size - 1)?;
                let trace_start = position_parameter.at(position);
                let trace_end = (trace_start.clone() + bucket_size)?;
                let start_dim: Dim = trace_start.into();
                let last_dim = last_parameter.at(real - 1).into();
                let graph = crate::capture(&[&position_parameter, &last_parameter], || {
                    let input = self.tokens.narrow(0, &start_dim, bucket_size)?;
                    forward(
                        &input,
                        bucket_size,
                        &last_dim,
                        &start_dim,
                        &trace_end,
                        &position_parameter,
                    )
                })?;
                if graph.result().shape().len() != 1 {
                    return Err(Error::loading(
                        "prefill forward must return rank-one logits",
                    ));
                }
                *captured = Some(CapturedPrefill { graph });
            }
            let graph = &captured
                .as_ref()
                .ok_or_else(|| Error::loading("prefill graph was not captured"))?
                .graph;
            graph.replay(&[position, real - 1])?;
            result = Some(graph.result().alias()?);
            consumed = consumed
                .checked_add(real)
                .ok_or_else(|| Error::loading("prefill progress overflowed"))?;
            position = position
                .checked_add(real)
                .ok_or_else(|| Error::loading("prefill position overflowed"))?;
        }
        result.ok_or_else(|| Error::loading("prefill produced no logits"))
    }

    /// Runs the padded chunk plan without capturing or replaying graphs.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, malformed logits, or a refused tensor
    /// operation.
    pub fn lazy(
        &self,
        tokens: &Tensor<u32>,
        start: u32,
        forward: impl FnMut(&Tensor<u32>, u32, &Dim, &Dim, &Dim, u32) -> Result<Tensor<T>>,
    ) -> Result<Tensor<T>> {
        run_lazy_chunked_prefill(tokens, start, self.chunk, &self.tokens, forward)
    }
}

/// Records the same fixed-size prefill chunks without capturing graphs.
///
/// Remainders use the same padded bucket plan as [`ChunkedPrefill`]. `forward`
/// must return the last real token's logits.
///
/// # Errors
///
/// Returns an error for invalid input, malformed logits, or a refused tensor
/// operation.
pub fn lazy_chunked_prefill<T: Element>(
    tokens: &Tensor<u32>,
    start: u32,
    chunk: u32,
    forward: impl FnMut(&Tensor<u32>, u32, &Dim, &Dim, &Dim, u32) -> Result<Tensor<T>>,
) -> Result<Tensor<T>> {
    let [sequence] = tokens
        .shape()
        .try_into()
        .map_err(|_| Error::loading("prefill tokens must have rank one"))?;
    let capacity = start
        .checked_add(
            padded_prefill_len(sequence, chunk)
                .ok_or_else(|| Error::loading("prefill capacity overflowed"))?,
        )
        .ok_or_else(|| Error::loading("prefill capacity overflowed"))?;
    let retained = Tensor::zeros(&[capacity])?;
    run_lazy_chunked_prefill(tokens, start, chunk, &retained, forward)
}

fn run_lazy_chunked_prefill<T: Element>(
    tokens: &Tensor<u32>,
    start: u32,
    chunk: u32,
    retained: &Tensor<u32>,
    mut forward: impl FnMut(&Tensor<u32>, u32, &Dim, &Dim, &Dim, u32) -> Result<Tensor<T>>,
) -> Result<Tensor<T>> {
    let [sequence] = tokens
        .shape()
        .try_into()
        .map_err(|_| Error::loading("prefill tokens must have rank one"))?;
    let values = tokens.to_vec()?;
    let plan = prefill_plan(sequence, chunk)?;
    retained.write(&staged_tokens(&values, start, retained.shape()[0])?)?;
    let mut consumed = 0_u32;
    let mut position = start;
    let mut result = None;
    for (real, bucket_size) in plan {
        let input = retained.narrow(0, position, bucket_size)?;
        let end = position
            .checked_add(bucket_size)
            .ok_or_else(|| Error::loading("prefill position overflowed"))?;
        let logits = forward(
            &input,
            bucket_size,
            &(real - 1).into(),
            &position.into(),
            &end.into(),
            position,
        )?;
        if logits.shape().len() != 1 {
            return Err(Error::loading(
                "prefill forward must return rank-one logits",
            ));
        }
        result = Some(logits);
        consumed = consumed
            .checked_add(real)
            .ok_or_else(|| Error::loading("prefill progress overflowed"))?;
        position = position
            .checked_add(real)
            .ok_or_else(|| Error::loading("prefill position overflowed"))?;
        if consumed < sequence {
            crate::eval()?;
        }
    }
    crate::eval()?;
    result.ok_or_else(|| Error::loading("prefill produced no logits"))
}

fn padded_prefill_len(sequence: u32, chunk: u32) -> Option<u32> {
    let full = sequence / chunk;
    let remainder = sequence % chunk;
    full.checked_mul(chunk)?.checked_add(if remainder == 0 {
        0
    } else {
        prefill_bucket(remainder, chunk).ok()?
    })
}

fn staged_tokens(values: &[u32], start: u32, capacity: u32) -> Result<Vec<u32>> {
    let begin =
        usize::try_from(start).map_err(|_| Error::loading("prefill offset does not fit usize"))?;
    let end = usize::try_from(
        start
            .checked_add(
                u32::try_from(values.len())
                    .map_err(|_| Error::loading("prefill length does not fit u32"))?,
            )
            .ok_or_else(|| Error::loading("prefill offset overflowed"))?,
    )
    .map_err(|_| Error::loading("prefill end does not fit usize"))?;
    let mut staged = vec![
        0_u32;
        usize::try_from(capacity)
            .map_err(|_| Error::loading("prefill capacity does not fit usize"))?
    ];
    staged
        .get_mut(begin..end)
        .ok_or_else(|| Error::loading("prefill token range is invalid"))?
        .copy_from_slice(values);
    Ok(staged)
}

fn prefill_bucket(sequence: u32, chunk: u32) -> Result<u32> {
    if sequence == 0 || sequence > chunk {
        return Err(Error::loading("prefill chunk length is invalid"));
    }
    if sequence == chunk {
        return Ok(chunk);
    }
    Ok(sequence.max(MIN_PREFILL_BUCKET).next_power_of_two())
}

fn prefill_plan(sequence: u32, chunk: u32) -> Result<Vec<(u32, u32)>> {
    if sequence == 0 {
        return Err(Error::loading("prefill tokens cannot be empty"));
    }
    let mut remaining = sequence;
    let mut plan = Vec::new();
    while remaining > 0 {
        let real = remaining.min(chunk);
        plan.push((real, prefill_bucket(real, chunk)?));
        remaining -= real;
    }
    Ok(plan)
}

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

impl<T: FloatElement> KvCache<T> {
    /// Normalizes projected Q/K rows, applies rotary embeddings, and writes K/V cache slots.
    ///
    /// The returned tensors are head-major query and cache prefixes for attention.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible projection shapes, cache ranges, parameters, or refused
    /// work.
    #[allow(clippy::too_many_arguments)]
    pub fn append_projected(
        &mut self,
        qkv: &Tensor<T>,
        norm: &Tensor<T>,
        positions: &Tensor<T>,
        query_heads: u32,
        eps: f32,
        theta: f32,
        start: &Dim,
        end: &Dim,
    ) -> Result<(Tensor<T>, Tensor<T>, Tensor<T>)> {
        let [sequence, total_heads, width]: [u32; 3] = qkv
            .shape()
            .try_into()
            .map_err(|_| Error::loading("QKV projection must have rank three"))?;
        let kv_heads = total_heads
            .checked_sub(query_heads)
            .and_then(|heads| heads.checked_div(2))
            .filter(|&heads| query_heads + heads * 2 == total_heads)
            .ok_or_else(|| Error::loading("QKV projection head counts are incompatible"))?;
        if self.key.shape()[0] != kv_heads {
            return Err(Error::loading("QKV projection does not match the cache"));
        }
        let norm_heads = query_heads
            .checked_add(kv_heads)
            .ok_or_else(|| Error::loading("QKV normalization head count overflowed"))?;
        if norm.shape() != [norm_heads, width] || positions.shape() != [sequence] {
            return Err(Error::loading(
                "QKV normalization or position shape is incompatible",
            ));
        }
        let kernel = qk_norm_rope_kernel(crate::sys::dtype(T::DTYPE)?, width, eps, theta)?;
        let query = qk_norm_rope(
            &kernel,
            &qkv.narrow(1, 0, query_heads)?,
            &norm.narrow(0, 0, query_heads)?,
            positions,
        )?
        .permute(&[1, 0, 2])?;
        let key = qk_norm_rope(
            &kernel,
            &qkv.narrow(1, query_heads, kv_heads)?,
            &norm.narrow(0, query_heads, kv_heads)?,
            positions,
        )?
        .permute(&[1, 0, 2])?;
        let value = qkv.narrow(1, norm_heads, kv_heads)?.permute(&[1, 0, 2])?;
        let (key, value) = self.append(&key, &value, start, sequence, end)?;
        Ok((query, key, value))
    }
}

fn attend_cached<T: Element>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    scale: f32,
    start: &Dim,
) -> Result<Tensor<T>> {
    let [query_heads, sequence, width]: [u32; 3] = query
        .shape()
        .try_into()
        .map_err(|_| Error::loading("attention query must have rank three"))?;
    let hidden = query_heads
        .checked_mul(width)
        .ok_or_else(|| Error::loading("attention output width overflowed"))?;
    let output = Tensor::<T>::empty(vec![sequence, query_heads, width])?;
    sdpa_into(
        query,
        key,
        value,
        &output.permute(&[1, 0, 2])?,
        scale,
        true,
        start,
    )?;
    output.reshape(&[sequence, hidden])
}

/// Applies normalized rotary Q/K projection, cache insertion, and cached decode attention.
///
/// # Errors
///
/// Returns an error for incompatible projections, cache ranges, parameters, or refused work.
#[allow(clippy::too_many_arguments)]
pub fn projected_cached_attention<T: FloatElement>(
    qkv: &Tensor<T>,
    norm: &Tensor<T>,
    positions: &Tensor<T>,
    cache: &mut KvCache<T>,
    query_heads: u32,
    eps: f32,
    theta: f32,
    scale: f32,
    start: &Dim,
    end: &Dim,
) -> Result<Tensor<T>> {
    let (query, key, value) =
        cache.append_projected(qkv, norm, positions, query_heads, eps, theta, start, end)?;
    attend_cached(&query, &key, &value, scale, start)
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
    cached_attention_selected(query, key, value, cache, scale, start, end, None)
}

/// Applies cached causal attention with an explicit Metal algorithm selection.
///
/// # Errors
///
/// Returns an error for incompatible shapes, cache ranges, rule context, or a refused selection.
#[allow(clippy::too_many_arguments)]
pub fn cached_attention_with<'a, T: Element>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    cache: &mut KvCache<T>,
    scale: f32,
    start: &Dim,
    end: &Dim,
    selection: impl Into<crate::target::metal::VariantChoice<'a>>,
) -> Result<Tensor<T>> {
    cached_attention_selected(
        query,
        key,
        value,
        cache,
        scale,
        start,
        end,
        Some(selection.into()),
    )
}

#[allow(clippy::too_many_arguments)]
fn cached_attention_selected<T: Element>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    cache: &mut KvCache<T>,
    scale: f32,
    start: &Dim,
    end: &Dim,
    selection: Option<crate::target::metal::VariantChoice<'_>>,
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
    if sequence != 1 {
        let hidden = query_heads
            .checked_mul(width)
            .ok_or_else(|| Error::loading("attention output width overflowed"))?;
        let attended = if let Some(selection) = selection {
            super::ops::sdpa_with(
                &query,
                &cached_key,
                &cached_value,
                scale,
                true,
                start,
                selection,
            )?
        } else {
            sdpa(&query, &cached_key, &cached_value, scale, true, start)?
        };
        return attended
            .permute(&[1, 0, 2])?
            .contiguous()?
            .reshape(&[sequence, hidden]);
    }
    if let Some(selection) = selection {
        let [query_heads, _, width]: [u32; 3] = query
            .shape()
            .try_into()
            .map_err(|_| Error::loading("attention query must have rank three"))?;
        let hidden = query_heads
            .checked_mul(width)
            .ok_or_else(|| Error::loading("attention output width overflowed"))?;
        let output = Tensor::<T>::empty(vec![1, query_heads, width])?;
        sdpa_into_with(
            &query,
            &cached_key,
            &cached_value,
            &output.permute(&[1, 0, 2])?,
            scale,
            true,
            start,
            selection,
        )?;
        output.reshape(&[1, hidden])
    } else {
        attend_cached(&query, &cached_key, &cached_value, scale, start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_plan_uses_fixed_chunks_and_power_of_two_remainders() {
        let cases = [
            (1, vec![(1, 16)]),
            (7, vec![(7, 16)]),
            (16, vec![(16, 16)]),
            (33, vec![(33, 64)]),
            (511, vec![(511, 512)]),
            (512, vec![(512, 512)]),
            (513, vec![(512, 512), (1, 16)]),
            (1_500, vec![(512, 512), (512, 512), (476, 512)]),
            (
                4_000,
                vec![
                    (512, 512),
                    (512, 512),
                    (512, 512),
                    (512, 512),
                    (512, 512),
                    (512, 512),
                    (512, 512),
                    (416, 512),
                ],
            ),
        ];
        for (sequence, expected) in cases {
            assert_eq!(
                prefill_plan(sequence, DEFAULT_PREFILL_CHUNK).unwrap(),
                expected,
                "sequence {sequence}"
            );
        }
    }

    #[test]
    fn padded_length_accounts_for_only_the_remainder_bucket() {
        for (sequence, padded) in [
            (1, 16),
            (7, 16),
            (16, 16),
            (33, 64),
            (511, 512),
            (512, 512),
            (513, 528),
            (1_500, 1_536),
            (4_000, 4_096),
        ] {
            assert_eq!(
                padded_prefill_len(sequence, DEFAULT_PREFILL_CHUNK),
                Some(padded),
                "sequence {sequence}"
            );
        }
    }

    #[test]
    fn staged_tokens_preserve_the_prompt_offset_and_padding() {
        assert_eq!(
            staged_tokens(&[7, 11, 13], 2, 8).unwrap(),
            [0, 0, 7, 11, 13, 0, 0, 0]
        );
        assert!(staged_tokens(&[7, 11, 13], 6, 8).is_err());
    }
}
