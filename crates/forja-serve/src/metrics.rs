use serde::Serialize;
use std::{fmt::Write as _, time::Duration};
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
    pub(crate) scheduler_time: [u64; 32],
    pub(crate) queue_time: [u64; 32],
    pub(crate) parser_mask_time: [u64; 32],
    pub(crate) sse_time: [u64; 32],
    pub(crate) gpu_time_ns: u64,
    pub(crate) live_bytes: u64,
    pub(crate) pool_bytes: u64,
    pub(crate) table_bytes: u64,
    pub(crate) slot_bytes: u64,
    pub(crate) graph_bytes: u64,
    pub(crate) scratch_bytes: u64,
}

impl ServingMetrics {
    /// Renders current bounded aggregates in Prometheus text format.
    #[must_use]
    pub fn prometheus(&self, live: bool, ready: bool) -> String {
        let mut output = String::new();
        gauge(&mut output, "forja_live", live);
        gauge(&mut output, "forja_ready", ready);
        counter(&mut output, "forja_prefix_requests_total", self.requests);
        counter(&mut output, "forja_prefix_hits_total", self.hits);
        counter(&mut output, "forja_prefix_misses_total", self.misses);
        counter(
            &mut output,
            "forja_prefix_reused_tokens_total",
            self.reused_tokens,
        );
        counter(
            &mut output,
            "forja_prefix_invalidations_total",
            self.invalidations,
        );
        counter(&mut output, "forja_prefix_rewinds_total", self.rewinds);
        histogram(&mut output, "forja_prefix_match_seconds", &self.match_time);
        histogram(
            &mut output,
            "forja_prefix_rewind_seconds",
            &self.rewind_time,
        );
        histogram(
            &mut output,
            "forja_suffix_prefill_seconds",
            &self.suffix_prefill_time,
        );
        histogram(&mut output, "forja_ttft_seconds", &self.ttft);
        histogram(
            &mut output,
            "forja_uncached_ttft_seconds",
            &self.uncached_ttft,
        );
        histogram(&mut output, "forja_scheduler_seconds", &self.scheduler_time);
        histogram(&mut output, "forja_queue_seconds", &self.queue_time);
        histogram(
            &mut output,
            "forja_parser_mask_seconds",
            &self.parser_mask_time,
        );
        histogram(&mut output, "forja_sse_seconds", &self.sse_time);
        gauge_u64(
            &mut output,
            "forja_engine_gpu_time_nanoseconds",
            self.gpu_time_ns,
        );
        gauge_u64(&mut output, "forja_engine_live_bytes", self.live_bytes);
        gauge_u64(&mut output, "forja_pool_bytes", self.pool_bytes);
        gauge_u64(&mut output, "forja_table_bytes", self.table_bytes);
        gauge_u64(&mut output, "forja_slot_bytes", self.slot_bytes);
        gauge_u64(&mut output, "forja_graph_bytes", self.graph_bytes);
        gauge_u64(&mut output, "forja_scratch_bytes", self.scratch_bytes);
        output
    }
}

fn gauge(output: &mut String, name: &str, value: bool) {
    output.push_str("# TYPE ");
    output.push_str(name);
    output.push_str(" gauge\n");
    output.push_str(name);
    output.push(' ');
    output.push(if value { '1' } else { '0' });
    output.push('\n');
}

fn counter(output: &mut String, name: &str, value: u64) {
    output.push_str("# TYPE ");
    output.push_str(name);
    output.push_str(" counter\n");
    output.push_str(name);
    output.push(' ');
    output.push_str(&value.to_string());
    output.push('\n');
}

fn gauge_u64(output: &mut String, name: &str, value: u64) {
    output.push_str("# TYPE ");
    output.push_str(name);
    output.push_str(" gauge\n");
    output.push_str(name);
    output.push(' ');
    output.push_str(&value.to_string());
    output.push('\n');
}

fn histogram(output: &mut String, name: &str, buckets: &[u64; 32]) {
    output.push_str("# TYPE ");
    output.push_str(name);
    output.push_str(" histogram\n");
    let mut cumulative = 0_u64;
    for (power, count) in buckets.iter().enumerate() {
        cumulative = cumulative.saturating_add(*count);
        let micros = 1_u64 << power;
        output.push_str(name);
        output.push_str("_bucket{le=\"");
        let _ = write!(output, "{}.{:06}", micros / 1_000_000, micros % 1_000_000);
        output.push_str("\"} ");
        output.push_str(&cumulative.to_string());
        output.push('\n');
    }
    output.push_str(name);
    output.push_str("_bucket{le=\"+Inf\"} ");
    output.push_str(&cumulative.to_string());
    output.push('\n');
    output.push_str(name);
    output.push_str("_count ");
    output.push_str(&cumulative.to_string());
    output.push('\n');
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

    #[test]
    fn prometheus_text_has_only_bounded_metric_labels() {
        let mut metrics = ServingMetrics {
            requests: 7,
            hits: 3,
            gpu_time_ns: 11,
            ..ServingMetrics::default()
        };
        observe(&mut metrics.ttft, Duration::from_micros(33));
        observe(&mut metrics.scheduler_time, Duration::from_micros(7));
        let text = metrics.prometheus(true, false);
        assert!(text.contains("forja_live 1\n"));
        assert!(text.contains("forja_ready 0\n"));
        assert!(text.contains("forja_prefix_requests_total 7\n"));
        assert!(text.contains("forja_ttft_seconds_count 1\n"));
        assert!(text.contains("forja_scheduler_seconds_count 1\n"));
        assert!(text.contains("forja_queue_seconds_count 0\n"));
        assert!(text.contains("forja_engine_gpu_time_nanoseconds 11\n"));
        assert!(!text.contains("request_id"));
        assert!(!text.contains("path="));
    }
}
