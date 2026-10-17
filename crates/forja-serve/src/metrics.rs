use serde::Serialize;
use std::time::Duration;
/// Serializable aggregate measurements with log2-microsecond latency buckets.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ServingMetrics {
    pub(crate) requests: u64,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) reused_tokens: u64,
    pub(crate) invalidations: u64,
    pub(crate) rewinds: u64,
    pub(crate) match_time: [u64; 32],
    pub(crate) rewind_time: [u64; 32],
    pub(crate) suffix_prefill_time: [u64; 32],
    pub(crate) ttft: [u64; 32],
    pub(crate) uncached_ttft: [u64; 32],
}
pub(crate) fn observe(histogram: &mut [u64; 32], duration: Duration) {
    let micros = u64::try_from(duration.as_micros()).unwrap_or(u64::MAX);
    let bucket = usize::try_from(micros.max(1).ilog2())
        .unwrap_or(usize::MAX)
        .min(histogram.len() - 1);
    histogram[bucket] = histogram[bucket].saturating_add(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn latency_histograms_are_fixed_and_saturating() {
        let mut histogram = [0; 32];
        observe(&mut histogram, Duration::from_micros(1));
        observe(&mut histogram, Duration::MAX);
        assert_eq!((histogram[0], histogram[31]), (1, 1));
    }
}
