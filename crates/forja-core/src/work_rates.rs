//! Calibrated time estimates for checked, schedule-derived GPU work.

use std::time::Duration;

use crate::KernelCost;

/// Device-memory access pattern used to price scheduled traffic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryAccess {
    /// Adjacent accesses with independently calibrated read and copy rates.
    Coalesced,
    /// Strided or indexed accesses priced at the scattered-read rate.
    Scattered,
}

/// Positive finite device rates for GPU work admission, independent of model identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkRates {
    rates: [f64; 7],
}

impl WorkRates {
    /// Validates read/copy/gather bytes per second, matrix/scalar/SFU operations per second,
    /// and seconds per dependent launch. Rates must come from host-controlled calibration.
    #[must_use]
    pub fn new(
        read: f64,
        copy: f64,
        gather: f64,
        matrix: f64,
        scalar: f64,
        sfu: f64,
        launch: f64,
    ) -> Option<Self> {
        let rates = [read, copy, gather, matrix, scalar, sfu, launch];
        rates
            .iter()
            .all(|rate| rate.is_finite() && *rate > 0.0)
            .then_some(Self { rates })
    }

    /// Predicts GPU time by adding calibrated memory, arithmetic and launch costs.
    /// Scalar operations include SFUs; matrix FLOPs must exclude work counted as scalar.
    /// Returns `None` on inconsistent subsets, overflow, or zero launches.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn predict(
        self,
        cost: KernelCost,
        launches: u64,
        access: MemoryAccess,
    ) -> Option<Duration> {
        if launches == 0 {
            return None;
        }
        let [read, copy, gather, matrix, scalar, sfu, launch] = self.rates;
        let bytes = cost.read_bytes.checked_add(cost.write_bytes)?;
        let memory = match access {
            MemoryAccess::Coalesced => (cost.read_bytes as f64 / read).max(bytes as f64 / copy),
            MemoryAccess::Scattered => bytes as f64 / gather,
        };
        let scalar_ops = cost.scalar_ops.checked_sub(cost.sfu_ops)?;
        let seconds = memory
            + cost.flops as f64 / matrix
            + scalar_ops as f64 / scalar
            + cost.sfu_ops as f64 / sfu
            + launches as f64 * launch;
        Duration::try_from_secs_f64(seconds)
            .ok()
            .filter(|time| !time.is_zero())
    }

    /// Returns named rates and units for diagnostic provenance.
    #[must_use]
    pub fn json(self) -> serde_json::Value {
        let [read, copy, gather, matrix, scalar, sfu, launch] = self.rates;
        serde_json::json!({"read_bytes_per_second": read, "copy_bytes_per_second": copy,
            "gather_bytes_per_second": gather, "matrix_flops_per_second": matrix,
            "scalar_ops_per_second": scalar, "sfu_ops_per_second": sfu,
            "dependent_launch_seconds": launch})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admission_prices_each_resource_and_access_class() {
        let rates = WorkRates::new(100.0, 200.0, 10.0, 1000.0, 100.0, 10.0, 0.01).unwrap();
        let cost = KernelCost {
            read_bytes: 100,
            write_bytes: 100,
            flops: 1000,
            scalar_ops: 110,
            sfu_ops: 10,
            ..KernelCost::default()
        };
        assert_eq!(
            rates.predict(cost, 7, MemoryAccess::Coalesced),
            Some(Duration::from_secs_f64(4.07))
        );
        assert_eq!(
            rates.predict(cost, 7, MemoryAccess::Scattered),
            Some(Duration::from_secs_f64(23.07))
        );
        assert!(rates.predict(cost, 0, MemoryAccess::Coalesced).is_none());
        assert!(
            rates
                .predict(
                    KernelCost {
                        sfu_ops: 1,
                        ..KernelCost::default()
                    },
                    1,
                    MemoryAccess::Coalesced
                )
                .is_none()
        );
        assert!(
            rates
                .predict(
                    KernelCost {
                        read_bytes: u64::MAX,
                        write_bytes: 1,
                        ..KernelCost::default()
                    },
                    1,
                    MemoryAccess::Coalesced
                )
                .is_none()
        );
    }

    #[test]
    fn calibration_prices_critical_metadata_and_group_overhead() {
        let rate = WorkRates::new(100.0, 200.0, 10.0, 1000.0, 100.0, 10.0, 0.01).unwrap();
        let observation = WorkObservation {
            name: "probe".into(),
            kernel: "probe".into(),
            units: 1,
            gpu: Duration::from_secs(1),
        };
        assert!(
            WorkCalibration::new([rate; 3], 0.0, 128, 1.0, vec![observation.clone()]).is_none()
        );
        assert!(
            WorkCalibration::new([rate; 3], 1.0, 128, f64::NAN, vec![observation.clone()])
                .is_none()
        );
        assert!(WorkCalibration::new([rate; 3], 1.0, 128, 1.0, vec![]).is_none());
        let rates = WorkCalibration::new([rate; 3], 0.1, 128, 1.0, vec![observation]).unwrap();
        let critical = KernelCost {
            read_bytes: 2,
            metadata_read_bytes: 2,
            ..KernelCost::default()
        };
        assert_eq!(
            rates.predict(
                KernelCost::default(),
                critical,
                false,
                1,
                (7, 7 * 128),
                MemoryAccess::Coalesced
            ),
            Some(Duration::from_secs_f64(2.71))
        );
        assert_eq!(
            rates.predict(
                KernelCost::default(),
                critical,
                false,
                1,
                (1, 4096),
                MemoryAccess::Coalesced
            ),
            Some(Duration::from_secs_f64(5.21))
        );
        assert!(WorkCalibration::new([rate; 3], 0.1, 0, 1.0, rates.observations.clone()).is_none());
        let invalid = KernelCost {
            metadata_read_bytes: 3,
            ..critical
        };
        assert!(
            rates
                .predict(
                    critical,
                    invalid,
                    false,
                    1,
                    (1, 128),
                    MemoryAccess::Coalesced
                )
                .is_none()
        );
    }

    #[test]
    fn rates_refuse_nonfinite_and_unrepresentable_predictions() {
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            for index in 0..7 {
                let mut values = [1.0; 7];
                values[index] = invalid;
                let [read, copy, gather, matrix, scalar, sfu, launch] = values;
                assert!(WorkRates::new(read, copy, gather, matrix, scalar, sfu, launch).is_none());
            }
        }
        let rates = WorkRates::new(f64::MIN_POSITIVE, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0).unwrap();
        assert!(
            rates
                .predict(
                    KernelCost {
                        read_bytes: u64::MAX,
                        ..KernelCost::default()
                    },
                    1,
                    MemoryAccess::Coalesced
                )
                .is_none()
        );
    }
}

/// One host-controlled observation retained with admission predictions.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkObservation {
    /// Source probe and executed arithmetic configuration.
    pub name: String,
    /// Exact shader entry point.
    pub kernel: String,
    /// Executed units, excluding inactive lanes.
    pub units: u64,
    /// Observed device duration.
    pub gpu: Duration,
}

/// Immutable calibration supporting aggregate, serial and cooperative estimates.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkCalibration {
    rates: [WorkRates; 3],
    group_seconds: f64,
    group_threads: u64,
    dependent_read: f64,
    observations: Vec<WorkObservation>,
}
impl WorkCalibration {
    /// Validates rates for scheduling groups and pointer-dependent metadata reads.
    #[must_use]
    pub fn new(
        rates: [WorkRates; 3],
        group_seconds: f64,
        group_threads: u64,
        dependent_read: f64,
        observations: Vec<WorkObservation>,
    ) -> Option<Self> {
        (group_seconds.is_finite()
            && group_seconds > 0.0
            && group_threads > 0
            && dependent_read.is_finite()
            && dependent_read > 0.0
            && !observations.is_empty()
            && observations.iter().all(|s| s.units > 0 && !s.gpu.is_zero()))
        .then_some(Self {
            rates,
            group_seconds,
            group_threads,
            dependent_read,
            observations,
        })
    }
    /// Prices aggregate work and the longest serial/cooperative path independently.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn predict(
        &self,
        total: KernelCost,
        mut critical: KernelCost,
        cooperative: bool,
        launches: u64,
        geometry: (u64, u64),
        access: MemoryAccess,
    ) -> Option<Duration> {
        let aggregate = self.rates[0].predict(total, launches.max(1), access)?;
        critical.read_bytes = critical
            .read_bytes
            .checked_sub(critical.metadata_read_bytes)?;
        let metadata = critical.metadata_read_bytes as f64 / self.dependent_read;
        let critical = self.rates[if cooperative { 2 } else { 1 }].predict(
            critical,
            launches.max(1),
            access,
        )?;
        let (groups, threads) = geometry;
        let units = groups.max(threads.div_ceil(self.group_threads));
        let groups = Duration::try_from_secs_f64(units as f64 * self.group_seconds).ok()?;
        let critical = critical.checked_add(Duration::try_from_secs_f64(metadata).ok()?)?;
        aggregate.max(critical).checked_add(groups)
    }
    /// Retains rates, safety factors and raw observations in diagnostic records.
    #[must_use]
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"kind":"admission calibration, not model throughput",
            "rate_policy":"half the slowest independent observation; twice launch/group latency",
            "aggregate":self.rates[0].json(),"serial":self.rates[1].json(),
            "cooperative":self.rates[2].json(),"group_seconds":self.group_seconds, "group_reference_threads":self.group_threads,
            "dependent_read_bytes_per_second":self.dependent_read,
            "observations": self.observations.iter().map(|s| serde_json::json!({"name":s.name,
                "kernel":s.kernel,"units":s.units,"gpu_seconds":s.gpu.as_secs_f64()})).collect::<Vec<_>>()})
    }
}
