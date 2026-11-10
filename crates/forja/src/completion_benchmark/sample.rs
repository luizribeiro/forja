use forja_serve::DraftMetrics;
use serde::Serialize;
use serde_json::Value;

#[derive(Serialize)]
pub(super) struct Sample {
    pub repetition: u32,
    pub wall_seconds: f64,
    pub gpu_seconds: f64,
    pub measurement_window: [f64; 2],
    pub decode_wall_seconds: f64,
    pub verify_gpu_seconds: f64,
    pub plain_gpu_seconds: f64,
    pub prompt_tokens: Vec<u32>,
    pub cached_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub finish_reasons: Vec<String>,
}

#[derive(Serialize)]
pub(super) struct RowRecord {
    #[serde(flatten)]
    pub observations: DraftMetrics,
    pub mean_accepted_per_round: f64,
    pub verify_gpu_seconds: f64,
    pub identity: Value,
    pub prompt_source_sha256: String,
    pub k: u32,
    pub drafter: Value,
    pub token_stream_hashes: Vec<String>,
    pub minimum_speedup: f64,
    pub finish_reasons: std::collections::BTreeMap<String, u64>,
    pub output_counts: std::collections::BTreeMap<u32, u64>,
    pub samples: Vec<Sample>,
    pub warmup: super::warmup::Warmup,
}

pub(super) fn finish_name(reason: forja_serve::FinishReason) -> &'static str {
    match reason {
        forja_serve::FinishReason::Stop => "stop",
        forja_serve::FinishReason::Length => "length",
        forja_serve::FinishReason::ToolCalls => "tool_calls",
    }
}

pub(super) fn validate_usage(
    usage: forja_serve::Usage,
    reason: forja_serve::FinishReason,
    prompt: usize,
    output: u32,
) -> Result<(), String> {
    if usize::try_from(usage.prompt_tokens).ok() != Some(prompt) {
        return Err("benchmark prompt tokens differ from service usage".to_owned());
    }
    if usage.completion_tokens != output || reason != forja_serve::FinishReason::Length {
        return Err(
            "degenerate completion: full requested output and length finish required".to_owned(),
        );
    }
    Ok(())
}

pub(super) fn counts<K: Ord>(
    values: impl Iterator<Item = K>,
) -> std::collections::BTreeMap<K, u64> {
    let mut counts = std::collections::BTreeMap::new();
    for value in values {
        let count = counts.entry(value).or_insert(0_u64);
        *count = count.saturating_add(1);
    }
    counts
}

#[derive(Clone, Copy)]
pub(super) struct RowContext<'a> {
    pub options: &'a super::Options,
    pub inputs: &'a Value,
    pub row: &'a super::workload::Row,
    pub prompt: &'a str,
    pub prompt_tokens: usize,
}

impl RowRecord {
    pub fn new(
        context: RowContext<'_>,
        observations: DraftMetrics,
        samples: Vec<Sample>,
        hashes: Vec<String>,
        warmup: super::warmup::Warmup,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        use super::record;
        use golden_fixtures::sha256_file;
        use serde_json::json;
        use std::time::Duration;
        let RowContext {
            options,
            inputs,
            row,
            prompt,
            prompt_tokens,
        } = context;
        let policy = &inputs["lookup_policy"];
        if policy["max-drafts"]
            .as_u64()
            .is_none_or(|cap| cap == 0 || cap > u64::from(u32::MAX))
        {
            return Err("missing or invalid max-drafts".into());
        }
        Ok(Self {
            mean_accepted_per_round: observations.mean_accepted(),
            verify_gpu_seconds: Duration::from_nanos(observations.verify_gpu_ns).as_secs_f64(),
            identity: json!({"name":row.name,"group":row.group,"seed":row.seed,"temperature":row.temperature,
            "top_k":0,"top_p":1.0,"prompt_sha256":record::prompt_hash(prompt),"prompt_tokens":prompt_tokens,
            "output_tokens":row.output_tokens,"concurrency":row.concurrency,"reps":options.reps}),
            prompt_source_sha256: sha256_file(
                options
                    .suite
                    .parent()
                    .ok_or("suite has no parent")?
                    .join(&row.prompt),
            )?,
            k: 0,
            drafter: json!({"identity":"none", "policy":policy,"policy_sha256":record::hash(&policy)?}),
            warmup,
            observations,
            token_stream_hashes: hashes,
            minimum_speedup: row.minimum_speedup,
            finish_reasons: counts(
                samples
                    .iter()
                    .flat_map(|sample| sample.finish_reasons.iter().cloned()),
            ),
            output_counts: counts(
                samples
                    .iter()
                    .flat_map(|sample| sample.output_tokens.iter().copied()),
            ),
            samples,
        })
    }
}

#[cfg(test)]
mod tests {
    use forja_serve::{FinishReason, Usage};
    #[test]
    fn full_length_streams_require_matching_prompt_usage() {
        assert_eq!(super::finish_name(FinishReason::Length), "length");
        assert_eq!(super::finish_name(FinishReason::Stop), "stop");
        assert_eq!(super::finish_name(FinishReason::ToolCalls), "tool_calls");
        let usage = Usage {
            prompt_tokens: 4097,
            cached_tokens: 0,
            completion_tokens: 256,
        };
        assert!(super::validate_usage(usage, FinishReason::Length, 4097, 256).is_ok());
        assert!(super::validate_usage(usage, FinishReason::Length, 4096, 256).is_err());
        assert!(super::validate_usage(usage, FinishReason::Stop, 4097, 256).is_err());
        assert!(
            super::validate_usage(
                Usage {
                    completion_tokens: 7,
                    ..usage
                },
                FinishReason::Stop,
                4097,
                256
            )
            .is_err()
        );
    }
}
