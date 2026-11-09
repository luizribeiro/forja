use super::{Mode, record};
use forja_serve::DraftMetrics;
use serde::Serialize;
use serde_json::Value;
use std::error::Error;

#[derive(Serialize)]
pub(super) struct DraftState {
    pub k: u32,
    pub drafting: &'static str,
    pub drafter: Drafter,
}

#[derive(Serialize)]
pub(super) struct Drafter {
    identity: &'static str,
    policy: Value,
    policy_sha256: String,
}

impl DraftState {
    pub fn new(metrics: &DraftMetrics, mode: Mode, policy: &Value) -> Result<Self, Box<dyn Error>> {
        let maximum = policy["max-drafts"]
            .as_u64()
            .filter(|maximum| *maximum > 0)
            .ok_or("missing or invalid max-drafts")?;
        let k = metrics
            .k_eff
            .iter()
            .filter(|(_, count)| **count > 0)
            .map(|(&k, _)| k)
            .max()
            .unwrap_or(0);
        let entered = metrics.rounds > 0;
        if u64::from(k) > maximum || entered != (k > 0) || (mode == Mode::Plain && entered) {
            return Err("inconsistent observed draft depth".into());
        }
        let identity = if entered {
            match metrics
                .episode_entry_reasons
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>()
                .as_slice()
            {
                ["prompt-lookup-match"] => "host-prompt-lookup",
                ["external-proposal"] => "external",
                _ => return Err("missing or ambiguous observed drafter source".into()),
            }
        } else {
            "none"
        };
        Ok(Self {
            k,
            drafting: if mode == Mode::Plain {
                "disabled"
            } else if entered {
                "entered"
            } else {
                "never-entered"
            },
            drafter: Drafter {
                identity,
                policy: policy.clone(),
                policy_sha256: record::hash(policy)?,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reported_depth_and_source_require_observed_rounds() {
        let policy = serde_json::json!({"max-drafts":2});
        let mut metrics = DraftMetrics::default();
        let never = DraftState::new(&metrics, Mode::Draft, &policy).unwrap();
        assert_eq!(never.k, 0);
        assert_eq!(never.drafting, "never-entered");
        assert_eq!(never.drafter.identity, "none");
        metrics.rounds = 1;
        metrics.k_eff.insert(1, 1);
        metrics
            .episode_entry_reasons
            .insert("prompt-lookup-match".to_owned(), 1);
        let observed = DraftState::new(&metrics, Mode::Draft, &policy).unwrap();
        assert_eq!(observed.k, 1);
        assert_eq!(observed.drafter.identity, "host-prompt-lookup");
        assert!(DraftState::new(&metrics, Mode::Plain, &policy).is_err());
        assert!(DraftState::new(&metrics, Mode::Draft, &serde_json::json!({})).is_err());
    }
}
