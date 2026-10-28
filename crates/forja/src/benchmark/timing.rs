use super::{Duration, Error};

#[derive(Clone, Copy)]
pub(super) enum Stage {
    Prefix,
    Prime,
    Setup,
    Measured,
    Probe,
}

pub(super) struct Timings {
    seconds: [f64; 5],
}

impl Timings {
    pub(super) const fn new() -> Self {
        Self { seconds: [0.0; 5] }
    }

    pub(super) fn add(&mut self, stage: Stage, elapsed: Duration) {
        self.seconds[stage as usize] += elapsed.as_secs_f64();
    }

    pub(super) fn report(&self, elapsed: Duration) -> Result<serde_json::Value, Box<dyn Error>> {
        let total = elapsed.as_secs_f64();
        let accounted: f64 = self.seconds.iter().sum();
        if accounted > total + 1e-6 {
            return Err("benchmark phase timers overlap".into());
        }
        Ok(serde_json::json!({
            "prefix_preparation_seconds": self.seconds[Stage::Prefix as usize],
            "priming_seconds": self.seconds[Stage::Prime as usize],
            "request_setup_seconds": self.seconds[Stage::Setup as usize],
            "measured_seconds": self.seconds[Stage::Measured as usize],
            "output_probe_seconds": self.seconds[Stage::Probe as usize],
            "other_seconds": (total - accounted).max(0.0),
            "total_seconds": total,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_accounting_preserves_unattributed_work_and_refuses_overlap() {
        let mut times = Timings::new();
        for stage in [
            Stage::Prefix,
            Stage::Prime,
            Stage::Setup,
            Stage::Measured,
            Stage::Probe,
        ] {
            times.add(stage, Duration::from_secs(1));
        }
        let report = times.report(Duration::from_secs(7)).unwrap();
        assert_eq!(report["other_seconds"], 2.0);
        assert_eq!(report["measured_seconds"], 1.0);
        assert!(times.report(Duration::from_secs(4)).is_err());
    }
}
