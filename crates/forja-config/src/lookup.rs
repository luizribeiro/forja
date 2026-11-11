use serde::{Deserialize, Deserializer, Serialize, de};

const MAX_SUFFIX: usize = 128;
const MAX_CONTEXT_ARMS: usize = 64;

/// Validated guest policy for suffix lookup over accepted request history.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct LookupPolicy(LookupData);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
struct LookupData {
    min_match: usize,
    max_suffix: usize,
    max_drafts: u32,
    k_rule: Vec<LookupCap>,
}

/// Inclusive context range and its maximum permitted draft count.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LookupCap {
    /// First pending-token position in this range.
    pub lo: u32,
    /// Last pending-token position in this range.
    pub hi: u32,
    /// Maximum drafts, with zero disabling lookup.
    pub k: u32,
}

impl LookupPolicy {
    /// Returns the inclusive range of suffix lengths searched.
    #[must_use]
    pub fn match_lengths(&self) -> std::ops::RangeInclusive<usize> {
        self.0.min_match..=self.0.max_suffix
    }

    /// Returns the draft cap at the pending token's fed position.
    #[must_use]
    pub fn cap(&self, position: u32) -> u32 {
        self.0
            .k_rule
            .iter()
            .find(|arm| position <= arm.hi)
            .map_or(0, |arm| arm.k.min(self.0.max_drafts))
    }

    /// Returns the exclusive end of the declared context range.
    #[must_use]
    pub fn context_limit(&self) -> u64 {
        self.0.k_rule.last().map_or(0, |arm| u64::from(arm.hi) + 1)
    }
}

impl<'de> Deserialize<'de> for LookupPolicy {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let data = LookupData::deserialize(deserializer)?;
        if data.min_match == 0 || data.min_match > data.max_suffix || data.max_suffix > MAX_SUFFIX {
            return Err(de::Error::custom(
                "lookup match lengths must satisfy 1 <= min-match <= max-suffix <= 128",
            ));
        }
        if data.max_drafts == 0 || data.k_rule.is_empty() || data.k_rule.len() > MAX_CONTEXT_ARMS {
            return Err(de::Error::custom(
                "lookup requires positive max-drafts and 1..=64 context arms",
            ));
        }
        let mut next = 0_u64;
        for arm in &data.k_rule {
            if u64::from(arm.lo) != next || arm.lo > arm.hi || arm.k > data.max_drafts {
                return Err(de::Error::custom(
                    "lookup context arms must be contiguous from zero with k <= max-drafts",
                ));
            }
            next = u64::from(arm.hi) + 1;
        }
        if next > u64::from(u32::MAX) {
            return Err(de::Error::custom(
                "lookup positions must leave u32::MAX available as an occurrence sentinel",
            ));
        }
        Ok(Self(data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn policy_caps_context_without_gaps_or_unbounded_match_work() {
        let data = json!({"min-match": 8, "max-suffix": 32, "max-drafts": 2,
            "k-rule": [{"lo": 0, "hi": 6, "k": 2}, {"lo": 7, "hi": 32, "k": 0}]});
        let policy: LookupPolicy = serde_json::from_value(data.clone()).unwrap();
        assert_eq!(policy.match_lengths(), 8..=32);
        assert_eq!((policy.cap(6), policy.cap(7), policy.cap(33)), (2, 0, 0));
        assert_eq!(policy.context_limit(), 33);
        for (field, value) in [
            ("min-match", 0),
            ("min-match", 33),
            ("max-suffix", MAX_SUFFIX + 1),
            ("max-drafts", 0),
        ] {
            let mut bad = data.clone();
            bad[field] = json!(value);
            assert!(serde_json::from_value::<LookupPolicy>(bad).is_err());
        }
        for arms in [
            json!([]),
            json!([{"lo": 1, "hi": 6, "k": 2}]),
            json!([{"lo": 0, "hi": 6, "k": 3}]),
            json!([{"lo": 0, "hi": u32::MAX, "k": 2}]),
            json!([{"lo": 0, "hi": 6, "k": 2}, {"lo": 6, "hi": 9, "k": 1}]),
        ] {
            let mut bad = data.clone();
            bad["k-rule"] = arms;
            assert!(serde_json::from_value::<LookupPolicy>(bad).is_err());
        }
    }
}
