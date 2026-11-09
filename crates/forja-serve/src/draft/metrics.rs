use std::collections::BTreeMap;

/// Request-local observations of the draft/verify path.
#[derive(Clone, Debug, Default, Eq, PartialEq, serde::Serialize)]
pub struct DraftMetrics {
    /// Completed verify rounds.
    pub rounds: u64,
    /// Rounds indexed by the exact number of accepted drafts (zero included).
    #[serde(rename = "acceptance_histogram_by_depth")]
    pub acceptance_by_depth: BTreeMap<u32, u64>,
    /// Committed decode results by effective draft count; zero denotes plain decode.
    #[serde(rename = "k_eff_distribution")]
    pub k_eff: BTreeMap<u32, u64>,
    /// Entries into drafting from the plain pipeline.
    pub episodes_entered: u64,
    /// Entries indexed by the source that supplied an admitted proposal.
    pub episode_entry_reasons: BTreeMap<String, u64>,
    /// Exits indexed by the event that ended the episode.
    pub episodes_exited: BTreeMap<String, u64>,
    /// Wall time from prompt completion through the terminal decode boundary.
    pub decode_wall_ns: u64,
    /// GPU time for completed verify calls, excluding plain and prompt work.
    pub verify_gpu_ns: u64,
    #[serde(skip)]
    pub(crate) episode_active: bool,
    #[serde(skip)]
    pub(crate) zero_acceptance: bool,
}

impl DraftMetrics {
    /// Returns accepted drafts per completed verify round, or zero without rounds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_accepted(&self) -> f64 {
        if self.rounds == 0 {
            return 0.0;
        }
        self.acceptance_by_depth
            .iter()
            .map(|(&depth, &count)| f64::from(depth) * count as f64)
            .sum::<f64>()
            / self.rounds as f64
    }

    pub(crate) fn enter(&mut self, reason: &str) {
        if !self.episode_active {
            self.episodes_entered = self.episodes_entered.saturating_add(1);
            let count = self
                .episode_entry_reasons
                .entry(reason.to_owned())
                .or_default();
            *count = count.saturating_add(1);
            self.episode_active = true;
        }
    }

    pub(crate) fn exit(&mut self, reason: &str) {
        if self.episode_active {
            let reason = if self.zero_acceptance && reason == "lookup-ended" {
                "zero-acceptance"
            } else {
                reason
            };
            let count = self.episodes_exited.entry(reason.to_owned()).or_default();
            *count = count.saturating_add(1);
            self.episode_active = false;
            self.zero_acceptance = false;
        }
    }

    pub(crate) fn round(&mut self, k_eff: u32, accepted: u32, gpu_ns: u64) {
        self.rounds = self.rounds.saturating_add(1);
        let count = self.acceptance_by_depth.entry(accepted).or_default();
        *count = count.saturating_add(1);
        self.decode(k_eff);
        self.verify_gpu_ns = self.verify_gpu_ns.saturating_add(gpu_ns);
        self.zero_acceptance = accepted == 0;
    }

    pub(crate) fn decode(&mut self, k_eff: u32) {
        let count = self.k_eff.entry(k_eff).or_default();
        *count = count.saturating_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terminal_cuts_keep_their_episode_exit_reason() {
        for reason in ["stop-or-eos", "max-tokens"] {
            let mut metrics = DraftMetrics::default();
            metrics.enter("prompt-lookup-match");
            metrics.round(1, 0, 7);
            metrics.exit(reason);
            assert_eq!(metrics.episodes_exited[reason], 1);
            assert!(!metrics.episodes_exited.contains_key("zero-acceptance"));
        }
    }

    #[test]
    fn rounds_and_episode_reasons_preserve_zero_acceptance() {
        let mut metrics = DraftMetrics::default();
        assert_eq!(metrics.mean_accepted(), 0.0);
        metrics.enter("prompt-lookup-match");
        metrics.enter("prompt-lookup-match");
        metrics.round(2, 2, 7);
        metrics.round(1, 0, 11);
        metrics.exit("lookup-ended");
        metrics.exit("finished");
        metrics.decode(0);
        assert_eq!(metrics.mean_accepted(), 1.0);
        assert_eq!(metrics.rounds, 2);
        assert_eq!(metrics.verify_gpu_ns, 18);
        assert_eq!(metrics.episodes_entered, 1);
        assert_eq!(metrics.episodes_exited["zero-acceptance"], 1);
        assert_eq!(metrics.k_eff, BTreeMap::from([(0, 1), (1, 1), (2, 1)]));
    }
}
