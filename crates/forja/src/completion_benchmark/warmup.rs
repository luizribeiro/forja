use forja_serve::{ChatService, CompletionRequest, CompletionSession};
use serde::Serialize;
use std::error::Error;

#[derive(Serialize)]
pub(super) struct Warmup {
    pub groups: u32,
    pub concurrency: u32,
    pub prompt_tokens_per_lane: usize,
    pub output_tokens_per_lane: u32,
    pub observed_rounds: u64,
    pub full_prompt_requests: u64,
}

pub(super) async fn run<S: ChatService>(
    session: &CompletionSession<S>,
    request: CompletionRequest,
    groups: u32,
    concurrency: u32,
    prompt_tokens: usize,
) -> Result<Warmup, Box<dyn Error>> {
    let mut observed_rounds = 0_u64;
    for _ in 0..groups {
        let requests = vec![request.clone(); usize::try_from(concurrency)?];
        for completion in session.complete_texts(requests).await? {
            observed_rounds = observed_rounds
                .checked_add(completion.drafting.rounds)
                .ok_or("warm-up round overflow")?;
        }
    }
    Ok(Warmup {
        groups,
        concurrency,
        prompt_tokens_per_lane: prompt_tokens,
        output_tokens_per_lane: request
            .params
            .max_tokens
            .ok_or("missing warm-up output budget")?,
        observed_rounds,
        full_prompt_requests: u64::from(groups)
            .checked_mul(u64::from(concurrency))
            .ok_or("warm-up count overflow")?,
    })
}

pub(super) fn output_tokens(
    policy: &serde_json::Value,
    requested: u32,
) -> Result<u32, Box<dyn Error>> {
    let minimum = u32::try_from(policy["min-match"].as_u64().ok_or("missing min-match")?)?;
    let drafts = u32::try_from(policy["max-drafts"].as_u64().ok_or("missing max-drafts")?)?;
    Ok(requested.max(
        minimum
            .checked_add(drafts)
            .and_then(|count| count.checked_add(2))
            .ok_or("warm-up token overflow")?,
    ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn lookup_initialization_preserves_full_output_budgets() {
        let policy = serde_json::json!({"min-match":8,"max-drafts":2});
        assert_eq!(super::output_tokens(&policy, 1).unwrap(), 12);
        assert_eq!(super::output_tokens(&policy, 256).unwrap(), 256);
        assert!(super::output_tokens(&serde_json::json!({"min-match":8}), 1).is_err());
        assert!(
            super::output_tokens(&serde_json::json!({"min-match":u32::MAX,"max-drafts":2}), 1)
                .is_err()
        );
    }
}
