use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
};

use forja_config::LookupPolicy;
use forja_host::SamplingParams;

use super::{DraftProposal, DraftSource};

/// Incremental suffix index for one append-only accepted request history.
///
/// Each accepted position owns one occurrence link and at most one fixed-size
/// hash entry. Catch-up is linear in new history; queries compare tokens exactly.
/// Proposes the continuation of the longest suffix among the most recent
/// `MAX_CHAIN_PROBES` occurrences of the min-match n-gram, preferring the most
/// recent occurrence on ties. Hash collisions share that budget and are rejected
/// by token comparisons. Older matches are ignored, which can lower acceptance.
/// A new request needs a fresh source.
pub struct PromptLookup {
    policy: LookupPolicy,
    indexed: usize,
    ends: HashMap<u64, u32>,
    previous: Vec<u32>,
}

impl PromptLookup {
    /// Creates an empty index under a validated guest policy.
    #[must_use]
    pub fn new(policy: LookupPolicy) -> Self {
        Self {
            policy,
            indexed: 0,
            ends: HashMap::new(),
            previous: Vec::new(),
        }
    }
}

// Repetitive n-grams must not make query work grow with accepted history.
const MAX_CHAIN_PROBES: usize = 64;
const NO_OCCURRENCE: u32 = u32::MAX;

fn ngram_hash(tokens: &[u32]) -> u64 {
    let mut hasher = DefaultHasher::new();
    tokens.hash(&mut hasher);
    hasher.finish()
}

fn retain_longer_match(prefix: &[u32], end: usize, maximum: usize, best: &mut (usize, usize)) {
    let length = prefix
        .iter()
        .rev()
        .zip(prefix[..end].iter().rev())
        .take(maximum)
        .take_while(|(left, right)| left == right)
        .count();
    if length > best.0 {
        *best = (length, end);
    }
}

impl PromptLookup {
    fn continuation(&self, prefix: &[u32]) -> Option<usize> {
        let minimum = *self.policy.match_lengths().start();
        let maximum = (*self.policy.match_lengths().end()).min(prefix.len());
        if prefix.len() < minimum {
            return None;
        }
        let mut next = *self
            .ends
            .get(&ngram_hash(&prefix[prefix.len() - minimum..]))?;
        let mut best = (minimum - 1, 0);
        for _ in 0..MAX_CHAIN_PROBES {
            if next == NO_OCCURRENCE {
                break;
            }
            let end = usize::try_from(next).ok()?;
            retain_longer_match(prefix, end, maximum, &mut best);
            next = self.previous[end];
            if best.0 == maximum {
                break;
            }
        }
        (best.0 >= minimum).then_some(best.1)
    }
}

impl DraftSource for PromptLookup {
    fn propose(&mut self, prefix: &[u32], position: u32, _: SamplingParams) -> DraftProposal {
        if prefix.len() < self.indexed {
            self.indexed = 0;
            self.ends.clear();
            self.previous.clear();
        }
        let length = u64::from(position) + 1;
        if u64::try_from(prefix.len()).ok() != Some(length) || length > self.policy.context_limit()
        {
            return DraftProposal::default();
        }
        let cap = self.policy.cap(position);
        if cap == 0 {
            return DraftProposal::default();
        }
        let minimum = *self.policy.match_lengths().start();
        for end in self.indexed..prefix.len() {
            let Ok(position) = u32::try_from(end) else {
                return DraftProposal::default();
            };
            let previous = if end >= minimum {
                self.ends
                    .insert(ngram_hash(&prefix[end - minimum..end]), position)
                    .unwrap_or(NO_OCCURRENCE)
            } else {
                NO_OCCURRENCE
            };
            self.previous.push(previous);
        }
        self.indexed = prefix.len();
        let tokens = self.continuation(prefix).map_or_else(Vec::new, |end| {
            prefix[end..]
                .iter()
                .copied()
                .take(usize::try_from(cap).unwrap_or(usize::MAX))
                .collect()
        });
        DraftProposal {
            tokens,
            rule_cap: cap,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source(minimum: usize, maximum: usize) -> PromptLookup {
        PromptLookup::new(
            serde_json::from_value(serde_json::json!({
                "min-match": minimum, "max-suffix": maximum, "max-drafts": 2,
                "k-rule": [{"lo": 0, "hi": u32::MAX - 1, "k": 2}]
            }))
            .unwrap(),
        )
    }

    fn probe(source: &mut PromptLookup, prefix: &[u32]) -> Vec<u32> {
        source
            .propose(
                prefix,
                u32::try_from(prefix.len() - 1).unwrap(),
                SamplingParams::default(),
            )
            .tokens
    }

    fn bounded_oracle(prefix: &[u32], minimum: usize, maximum: usize) -> Vec<u32> {
        if prefix.len() < minimum {
            return Vec::new();
        }
        let candidates: Vec<_> = (minimum..prefix.len())
            .rev()
            .filter(|end| prefix[end - minimum..*end] == prefix[prefix.len() - minimum..])
            .take(MAX_CHAIN_PROBES)
            .collect();
        (minimum..=maximum.min(prefix.len()))
            .rev()
            .find_map(|size| {
                candidates.iter().copied().find(|end| {
                    *end >= size && prefix[end - size..*end] == prefix[prefix.len() - size..]
                })
            })
            .map_or_else(Vec::new, |end| {
                prefix[end..].iter().take(2).copied().collect()
            })
    }

    #[test]
    fn longest_suffix_beats_a_more_recent_short_match() {
        let prefix: Vec<_> = (0..64).chain(32..40).chain([999]).chain(0..40).collect();
        assert_eq!(probe(&mut source(8, 8), &prefix), [999, 0]);
        assert_eq!(probe(&mut source(8, 32), &prefix), [40, 41]);
        assert_eq!(probe(&mut source(32, 32), &prefix), [40, 41]);
        assert_eq!(probe(&mut source(8, 32), &[1; 7]), [] as [u32; 0]);
        assert_eq!(probe(&mut source(8, 32), &[1; 8]), [] as [u32; 0]);
        assert_eq!(probe(&mut source(8, 32), &[1; 9]), [1]);
    }

    #[test]
    fn skipped_prefixes_and_repeated_probes_match_a_bounded_oracle() {
        let history: Vec<_> = (0..4097).map(|i| (i * i + i / 7) % 7).collect();
        for minimum in [1, 7, 8, 32, 33] {
            let mut lookup = source(minimum, minimum + 3);
            for length in [1, 7, 33, 97, 4097] {
                let prefix = &history[..length];
                let expected = bounded_oracle(prefix, minimum, minimum + 3);
                assert_eq!(probe(&mut lookup, prefix), expected);
                let entries = lookup.ends.len();
                assert_eq!(probe(&mut lookup, prefix), expected);
                assert_eq!(lookup.ends.len(), entries);
                assert_eq!(lookup.indexed, length);
                assert!(entries <= length.saturating_sub(minimum));
                assert_eq!(lookup.previous.len(), length);
            }
        }
    }

    #[test]
    fn occurrence_storage_is_constant_per_token_for_wide_suffix_searches() {
        let history: Vec<_> = (0..32768).collect();
        let mut wide = source(8, 32);
        let mut narrow = source(8, 8);
        for length in [7, 33, 4097, history.len()] {
            let _ = probe(&mut wide, &history[..length]);
            let _ = probe(&mut narrow, &history[..length]);
            assert_eq!(wide.previous.len(), length);
            assert!(wide.ends.len() <= length.saturating_sub(8));
            let payload =
                wide.previous.len() * size_of::<u32>() + wide.ends.len() * size_of::<(u64, u32)>();
            assert!(payload <= length * (size_of::<u32>() + size_of::<(u64, u32)>()));
            assert_eq!(wide.previous.capacity(), narrow.previous.capacity());
            assert_eq!(wide.ends.capacity(), narrow.ends.capacity());
        }
    }

    #[test]
    fn deep_chains_do_not_read_past_the_probe_budget() {
        let mut prefix: Vec<_> = (0..40).chain([999]).collect();
        for _ in 0..=MAX_CHAIN_PROBES {
            prefix.extend([777].into_iter().chain(32..40));
        }
        prefix.extend(0..40);
        let mut lookup = source(8, 32);
        let expected = bounded_oracle(&prefix, 8, 32);
        assert_eq!(expected, [0, 1]);
        assert_eq!(probe(&mut lookup, &prefix), expected);
        let mut end = lookup.ends[&ngram_hash(&prefix[prefix.len() - 8..])];
        for _ in 1..MAX_CHAIN_PROBES {
            end = lookup.previous[usize::try_from(end).unwrap()];
        }
        // An extra probe must fail on an out-of-range occurrence.
        lookup.previous[usize::try_from(end).unwrap()] = u32::MAX - 1;
        assert_eq!(probe(&mut lookup, &prefix), expected);
    }

    #[test]
    fn hash_collisions_cannot_produce_a_false_match_within_the_budget() {
        let prefix: Vec<_> = (0..40).chain([999]).chain(0..40).collect();
        let mut lookup = source(8, 32);
        let expected = bounded_oracle(&prefix, 8, 32);
        assert_eq!(expected, [999, 0]);
        assert_eq!(probe(&mut lookup, &prefix), expected);
        for (end, previous) in lookup.previous.iter_mut().enumerate() {
            *previous = end
                .checked_sub(1)
                .map_or(NO_OCCURRENCE, |end| u32::try_from(end).unwrap());
        }
        lookup.ends.clear();
        lookup.ends.insert(
            ngram_hash(&prefix[prefix.len() - 8..]),
            u32::try_from(prefix.len() - 1).unwrap(),
        );
        assert_eq!(probe(&mut lookup, &prefix), expected);
    }

    #[test]
    fn shrinking_history_rebuilds_occurrences_without_stale_positions() {
        let mut lookup = source(8, 32);
        assert_eq!(probe(&mut lookup, &[1; 33]), [1]);
        let prefix: Vec<_> = (0..7).collect();
        assert_eq!(
            probe(&mut lookup, &prefix),
            probe(&mut source(8, 32), &prefix)
        );
        assert_eq!(lookup.indexed, prefix.len());
        assert_eq!(lookup.previous.len(), prefix.len());
        assert_eq!(lookup.ends.len(), 0);
    }
}
