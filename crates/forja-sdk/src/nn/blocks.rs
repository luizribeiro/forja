//! Reusable transformer blocks.

#[cfg(target_family = "wasm")]
use crate::SamplingParams;
use crate::{Dim, Element, Error, Graph, Param, Result, Tensor};

use super::ops::sdpa;

/// Default number of tokens recorded in a full prefill graph.
pub const DEFAULT_PREFILL_CHUNK: u32 = 512;

const MIN_PREFILL_BUCKET: u32 = 16;

struct PrefillBucket<T: Element> {
    size: u32,
    captured: Option<CapturedPrefill<T>>,
}

struct CapturedPrefill<T: Element> {
    graph: Graph<Tensor<T>>,
}

/// Lazily captured prefill graphs and their retained token buffer.
pub struct ChunkedPrefill<T: Element> {
    max_context: u32,
    chunk: u32,
    tokens: Tensor<u32>,
    buckets: Vec<PrefillBucket<T>>,
}

impl<T: Element> ChunkedPrefill<T> {
    /// Allocates a retained token buffer and power-of-two buckets through `chunk`.
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
        let mut buckets = Vec::new();
        let mut size = MIN_PREFILL_BUCKET;
        loop {
            buckets.push(PrefillBucket {
                size,
                captured: None,
            });
            if size == chunk {
                break;
            }
            size = size
                .checked_mul(2)
                .ok_or_else(|| Error::loading("prefill bucket size overflowed"))?;
        }
        Ok(Self {
            max_context,
            chunk,
            tokens: Tensor::zeros(&[chunk])?,
            buckets,
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

    /// Writes and replays each chunk, capturing a missing bucket on first use.
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
        mut forward: impl FnMut(&Tensor<u32>, u32, &Dim, &Dim, &Dim) -> Result<Tensor<T>>,
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
        let mut consumed = 0_u32;
        let mut position = start;
        let mut result = None;
        for (real, bucket_size) in plan {
            let bucket_index = self
                .buckets
                .iter()
                .position(|bucket| bucket.size == bucket_size)
                .ok_or_else(|| Error::loading("prefill bucket was not allocated"))?;
            let padded = padded_tokens(&values, consumed, real, self.chunk)?;
            self.tokens.write(&padded)?;
            let bucket = &mut self.buckets[bucket_index];
            if bucket.captured.is_none() {
                let position_parameter = Param::new(0..=self.max_context - bucket_size)?;
                let last_parameter = Param::new(0..=bucket_size - 1)?;
                let trace_start = position_parameter.at(position);
                let trace_end = (trace_start.clone() + bucket_size)?;
                let start_dim = trace_start.into();
                let last_dim = last_parameter.at(real - 1).into();
                let input = self.tokens.narrow(0, 0, bucket_size)?;
                let graph = crate::capture(&[&position_parameter, &last_parameter], || {
                    forward(&input, bucket_size, &last_dim, &start_dim, &trace_end)
                })?;
                if graph.result().shape().len() != 1 {
                    return Err(Error::loading(
                        "prefill forward must return rank-one logits",
                    ));
                }
                bucket.captured = Some(CapturedPrefill { graph });
            }
            let graph = &bucket
                .captured
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
            if consumed < sequence {
                crate::eval()?;
            }
        }
        crate::eval()?;
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
        forward: impl FnMut(&Tensor<u32>, u32, &Dim, &Dim, &Dim) -> Result<Tensor<T>>,
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
    forward: impl FnMut(&Tensor<u32>, u32, &Dim, &Dim, &Dim) -> Result<Tensor<T>>,
) -> Result<Tensor<T>> {
    let retained = Tensor::zeros(&[chunk])?;
    run_lazy_chunked_prefill(tokens, start, chunk, &retained, forward)
}

fn run_lazy_chunked_prefill<T: Element>(
    tokens: &Tensor<u32>,
    start: u32,
    chunk: u32,
    retained: &Tensor<u32>,
    mut forward: impl FnMut(&Tensor<u32>, u32, &Dim, &Dim, &Dim) -> Result<Tensor<T>>,
) -> Result<Tensor<T>> {
    let [sequence] = tokens
        .shape()
        .try_into()
        .map_err(|_| Error::loading("prefill tokens must have rank one"))?;
    let values = tokens.to_vec()?;
    let plan = prefill_plan(sequence, chunk)?;
    let mut consumed = 0_u32;
    let mut position = start;
    let mut result = None;
    for (real, bucket_size) in plan {
        retained.write(&padded_tokens(&values, consumed, real, chunk)?)?;
        let input = retained.narrow(0, 0, bucket_size)?;
        let end = position
            .checked_add(bucket_size)
            .ok_or_else(|| Error::loading("prefill position overflowed"))?;
        let logits = forward(
            &input,
            bucket_size,
            &(real - 1).into(),
            &position.into(),
            &end.into(),
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

fn padded_tokens(values: &[u32], consumed: u32, real: u32, capacity: u32) -> Result<Vec<u32>> {
    let begin = usize::try_from(consumed)
        .map_err(|_| Error::loading("prefill offset does not fit usize"))?;
    let end = usize::try_from(
        consumed
            .checked_add(real)
            .ok_or_else(|| Error::loading("prefill offset overflowed"))?,
    )
    .map_err(|_| Error::loading("prefill end does not fit usize"))?;
    let mut padded = vec![
        0_u32;
        usize::try_from(capacity)
            .map_err(|_| Error::loading("prefill chunk does not fit usize"))?
    ];
    padded[..usize::try_from(real)
        .map_err(|_| Error::loading("prefill length does not fit usize"))?]
        .copy_from_slice(
            values
                .get(begin..end)
                .ok_or_else(|| Error::loading("prefill token range is invalid"))?,
        );
    Ok(padded)
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
}
