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
    pub decode_tokens_per_second: f64,
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
    #[serde(flatten)]
    pub draft: super::observed::DraftState,
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

pub(super) fn output_counts<'a>(
    requests: impl IntoIterator<Item = (u32, &'a [u32])>,
) -> Result<Vec<u32>, String> {
    requests
        .into_iter()
        .enumerate()
        .map(|(index, (reported, tokens))| {
            let actual = u32::try_from(tokens.len())
                .map_err(|_| format!("request {index}: token count exceeds u32"))?;
            if reported != actual {
                return Err(format!(
                    "request {index}: completion_tokens={reported}, token_ids={actual}"
                ));
            }
            Ok(actual)
        })
        .collect()
}

pub(super) fn validate_usage(
    requests: impl IntoIterator<Item = (forja_serve::Usage, forja_serve::FinishReason)>,
    prompt: usize,
    minimum: u32,
    maximum: u32,
) -> Result<(), String> {
    let requests = requests.into_iter().collect::<Vec<_>>();
    if requests.iter().all(|(usage, reason)| {
        usize::try_from(usage.prompt_tokens).ok() == Some(prompt)
            && crate::completion_stream::valid_output(
                u64::from(usage.completion_tokens),
                finish_name(*reason),
                u64::from(minimum),
                u64::from(maximum),
            )
    }) {
        return Ok(());
    }
    let actual = requests
        .iter()
        .enumerate()
        .map(|(index, (usage, reason))| {
            format!(
                "request {index}: output_tokens={}, finish_reason={}, prompt_tokens={}",
                usage.completion_tokens,
                finish_name(*reason),
                usage.prompt_tokens
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    Err(format!(
        "invalid completion: requested={maximum}, min_output_tokens={minimum}, expected_prompt_tokens={prompt}; {actual}"
    ))
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
        Ok(Self {
            mean_accepted_per_round: observations.mean_accepted(),
            verify_gpu_seconds: Duration::from_nanos(observations.verify_gpu_ns).as_secs_f64(),
            identity: json!({"name":row.name,"group":row.group,"seed":row.seed,"temperature":row.temperature,
            "top_k":0,"top_p":1.0,"prompt_sha256":record::prompt_hash(prompt),"prompt_tokens":prompt_tokens,
            "output_tokens":row.output_tokens,"min_output_tokens":row.min_output_tokens,"concurrency":row.concurrency,"reps":options.reps}),
            prompt_source_sha256: sha256_file(
                options
                    .suite
                    .parent()
                    .ok_or("suite has no parent")?
                    .join(&row.prompt),
            )?,
            draft: super::observed::DraftState::new(&observations, options.mode, policy)?,
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
    fn output_counts_require_usage_to_match_selected_tokens() {
        let tokens = [7, 8, 9];
        assert_eq!(
            super::output_counts([(3, tokens.as_slice()), (0, &[])]).unwrap(),
            [3, 0]
        );
        for reported in [2, 4] {
            let error =
                super::output_counts([(3, tokens.as_slice()), (reported, &tokens)]).unwrap_err();
            assert!(error.contains(&format!(
                "request 1: completion_tokens={reported}, token_ids=3"
            )));
        }
    }

    #[test]
    fn natural_stops_respect_the_declared_minimum() {
        let usage = |tokens| Usage {
            prompt_tokens: 4097,
            cached_tokens: 0,
            completion_tokens: tokens,
        };
        for (tokens, reason, valid) in [
            (128, FinishReason::Stop, true),
            (256, FinishReason::Stop, true),
            (256, FinishReason::Length, true),
            (127, FinishReason::Stop, false),
            (255, FinishReason::Length, false),
            (257, FinishReason::Stop, false),
            (256, FinishReason::ToolCalls, false),
        ] {
            assert_eq!(
                super::validate_usage([(usage(tokens), reason)], 4097, 128, 256).is_ok(),
                valid
            );
        }
        let requests = [
            (usage(256), FinishReason::Length),
            (usage(7), FinishReason::Stop),
        ];
        let error = super::validate_usage(requests, 4097, 128, 256).unwrap_err();
        assert!(error.contains("request 0: output_tokens=256, finish_reason=length"));
        assert!(error.contains("request 1: output_tokens=7, finish_reason=stop"));
        assert!(super::validate_usage(requests[..1].iter().copied(), 4096, 128, 256).is_err());
    }
}
