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
    /// Coalesced reads separated from arithmetic by synchronization phases.
    Phased,
    /// Repeated coalesced tile reads within a bounded live input footprint.
    Reused {
        /// Bytes of simultaneously live input data.
        footprint: u64,
    },
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

    /// Predicts GPU time with a shared twofold memory/arithmetic safety margin.
    /// Rates must be at most half measured rates: twice their maximum also bounds their serial sum.
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
        self.predict_with_margin(cost, launches, access, None)
    }

    #[allow(clippy::cast_precision_loss)]
    fn predict_with_margin(
        self,
        cost: KernelCost,
        launches: u64,
        access: MemoryAccess,
        serial_margin: Option<f64>,
    ) -> Option<Duration> {
        if launches == 0 {
            return None;
        }
        let [read, copy, gather, ..] = self.rates;
        let bytes = cost.read_bytes.checked_add(cost.write_bytes)?;
        let memory = match access {
            MemoryAccess::Coalesced | MemoryAccess::Reused { .. } | MemoryAccess::Phased => {
                (cost.read_bytes as f64 / read).max(bytes as f64 / copy)
            }
            MemoryAccess::Scattered => bytes as f64 / gather,
        };
        self.predict_memory(
            cost,
            launches,
            memory,
            matches!(access, MemoryAccess::Phased) || serial_margin.is_some(),
            serial_margin.map_or(1.0, |margin| margin / 2.0),
        )
    }

    #[allow(clippy::cast_precision_loss)]
    fn predict_reused(
        self,
        cost: KernelCost,
        launches: u64,
        footprint: u64,
        rate: f64,
    ) -> Option<Duration> {
        let unique = footprint.min(cost.read_bytes);
        let repeated = cost.read_bytes.checked_sub(unique)?;
        let [read, copy, ..] = self.rates;
        let cold = (unique as f64 / read).max(unique.checked_add(cost.write_bytes)? as f64 / copy);
        self.predict_memory(cost, launches, cold + repeated as f64 / rate, false, 1.0)
    }

    #[allow(clippy::cast_precision_loss)]
    fn predict_memory(
        self,
        cost: KernelCost,
        launches: u64,
        memory: f64,
        phased: bool,
        resource_scale: f64,
    ) -> Option<Duration> {
        if launches == 0 {
            return None;
        }
        let [_, _, _, matrix, scalar, sfu, launch] = self.rates;
        let scalar_ops = cost.scalar_ops.checked_sub(cost.sfu_ops)?;
        let arithmetic =
            cost.flops as f64 / matrix + scalar_ops as f64 / scalar + cost.sfu_ops as f64 / sfu;
        let resources = if phased {
            memory + arithmetic
        } else {
            memory.max(arithmetic)
        };
        let seconds = resources * resource_scale + launches as f64 * launch;
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
            Some(Duration::from_secs_f64(3.07))
        );
        assert_eq!(
            rates.predict(cost, 7, MemoryAccess::Scattered),
            Some(Duration::from_secs_f64(20.07))
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
    fn synchronization_phases_charge_memory_before_arithmetic() {
        let rates = WorkRates::new(100.0, 100.0, 10.0, 1000.0, 100.0, 10.0, 0.01).unwrap();
        let cost = KernelCost {
            read_bytes: 100,
            flops: 1000,
            ..KernelCost::default()
        };
        assert_eq!(
            rates.predict(cost, 1, MemoryAccess::Phased),
            Some(Duration::from_secs_f64(2.01))
        );
        assert_eq!(
            rates.predict(cost, 1, MemoryAccess::Coalesced),
            Some(Duration::from_secs_f64(1.01))
        );
    }

    #[test]
    fn shared_margin_still_bounds_serial_execution_at_observed_rates() {
        let rates = WorkRates::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0).unwrap();
        for memory in [1_u32, 7, 33] {
            for arithmetic in [1_u32, 7, 33] {
                let cost = KernelCost {
                    read_bytes: u64::from(memory),
                    flops: u64::from(arithmetic),
                    ..KernelCost::default()
                };
                let prediction = rates.predict(cost, 1, MemoryAccess::Coalesced).unwrap();
                assert!(prediction >= Duration::from_secs(u64::from(memory + arithmetic + 1)));
            }
        }
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
            WorkCalibration::new(
                [rate; 3],
                0.0,
                128,
                1.0,
                vec![observation.clone()],
                Duration::ZERO
            )
            .is_none()
        );
        assert!(
            WorkCalibration::new(
                [rate; 3],
                1.0,
                128,
                f64::NAN,
                vec![observation.clone()],
                Duration::ZERO
            )
            .is_none()
        );
        assert!(WorkCalibration::new([rate; 3], 1.0, 128, 1.0, vec![], Duration::ZERO).is_none());
        let rates =
            WorkCalibration::new([rate; 3], 0.1, 128, 1.0, vec![observation], Duration::ZERO)
                .unwrap();
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
        assert!(
            WorkCalibration::new(
                [rate; 3],
                0.1,
                0,
                1.0,
                rates.observations.clone(),
                Duration::ZERO
            )
            .is_none()
        );
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
    fn repeated_reads_pay_cold_traffic_and_respect_the_calibrated_footprint() {
        let rate = WorkRates::new(100.0, 100.0, 10.0, 1000.0, 1000.0, 1000.0, 0.01).unwrap();
        let observation = WorkObservation {
            name: "read".into(),
            kernel: "read".into(),
            units: 100,
            gpu: Duration::from_secs(1),
        };
        let calibration =
            WorkCalibration::new([rate; 3], 0.1, 128, 10.0, vec![observation], Duration::ZERO)
                .unwrap()
                .with_reused_read(100, 1000.0)
                .unwrap();
        let total = KernelCost {
            read_bytes: 200,
            ..KernelCost::default()
        };
        let predict = |access| {
            calibration
                .predict(total, KernelCost::default(), false, 1, (1, 1), access)
                .unwrap()
        };
        assert_eq!(
            predict(MemoryAccess::Reused { footprint: 100 }),
            Duration::from_secs_f64(1.21)
        );
        assert_eq!(
            predict(MemoryAccess::Reused { footprint: 101 }),
            predict(MemoryAccess::Coalesced)
        );
        assert_eq!(
            predict(MemoryAccess::Scattered),
            Duration::from_secs_f64(20.11)
        );
        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(calibration.clone().with_reused_read(100, invalid).is_none());
        }
        assert!(calibration.clone().with_reused_read(0, 100.0).is_none());
        let record = calibration.cache_record("read-source");
        let reused = WorkCalibration::from_cache_record(&record, "read-source").unwrap();
        assert_eq!(reused.reused_read, calibration.reused_read);
        let mut record: serde_json::Value = serde_json::from_str(&record).unwrap();
        record["calibration"]["reused_read"]["bytes_per_second"] = serde_json::json!(0);
        assert!(WorkCalibration::from_cache_record(&record.to_string(), "read-source").is_none());
    }

    #[test]
    fn cooperative_serial_margin_sums_resources_without_scaling_launches() {
        let rate = WorkRates::new(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 2.0).unwrap();
        let critical = KernelCost {
            read_bytes: 1,
            scalar_ops: 1,
            ..KernelCost::default()
        };
        assert_eq!(
            rate.predict_with_margin(critical, 1, MemoryAccess::Coalesced, Some(1.25)),
            Some(Duration::from_secs_f64(4.5))
        );
        let c = WorkCalibration::new(
            [rate; 3],
            0.1,
            128,
            1.0,
            vec![WorkObservation {
                name: "raw".into(),
                kernel: "raw".into(),
                units: 1,
                gpu: Duration::from_secs(1),
            }],
            Duration::ZERO,
        )
        .unwrap();
        for margin in [0.0, 0.5, f64::NAN, f64::INFINITY] {
            assert!(c.clone().with_serial_cooperative_margin(margin).is_none());
        }
        let c = c.with_serial_cooperative_margin(1.25).unwrap();
        let loaded =
            WorkCalibration::from_cache_record(&c.cache_record("source"), "source").unwrap();
        assert_eq!(loaded.serial_cooperative_margin, Some(1.25));
        assert_eq!(loaded.observations, c.observations);
    }

    #[test]
    fn cached_calibration_requires_exact_identity_and_valid_rates() {
        let rate = WorkRates::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.01).unwrap();
        let calibration = WorkCalibration::new(
            [rate; 3],
            0.1,
            128,
            10.0,
            vec![WorkObservation {
                name: "probe".into(),
                kernel: "kernel".into(),
                units: 7,
                gpu: Duration::from_millis(33),
            }],
            Duration::from_millis(7),
        )
        .unwrap()
        .with_launch_floor(Duration::from_millis(25))
        .unwrap();
        assert!(
            calibration
                .clone()
                .with_launch_floor(Duration::ZERO)
                .is_none()
        );
        assert_eq!(
            calibration.predict(
                KernelCost::default(),
                KernelCost::default(),
                false,
                2,
                (0, 0),
                MemoryAccess::Coalesced
            ),
            Some(Duration::from_millis(50))
        );
        let record = calibration.cache_record("device/source/OS");
        let reused = WorkCalibration::from_cache_record(&record, "device/source/OS").unwrap();
        assert_eq!(reused.observations, calibration.observations);
        assert_eq!(reused.rates, calibration.rates);
        assert_eq!(reused.command_buffer_reserve(), Duration::from_millis(7));
        assert_eq!(reused.launch_floor, calibration.launch_floor);
        assert_eq!(reused.json()["host_launch_floor_seconds"], 0.025);
        assert!(reused.reused);
        assert!(WorkCalibration::from_cache_record(&record, "device/source/other OS").is_none());
        assert!(WorkCalibration::from_cache_record(&record.replace("33", "0"), "other").is_none());
        let mut value: serde_json::Value = serde_json::from_str(&record).unwrap();
        value["calibration"]["serial"]["read_bytes_per_second"] = serde_json::json!(0);
        assert!(
            WorkCalibration::from_cache_record(&value.to_string(), "device/source/OS").is_none()
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
    reserve: Duration,
    reused_read: Option<(u64, f64)>,
    launch_floor: Option<Duration>,
    serial_cooperative_margin: Option<f64>,
    reused: bool,
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
        reserve: Duration,
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
            reserve,
            reused_read: None,
            launch_floor: None,
            serial_cooperative_margin: None,
            reused: false,
        })
    }
    /// Adds a measured coalesced reread rate, restricted to its calibrated footprint.
    #[must_use]
    pub fn with_reused_read(mut self, footprint: u64, bytes_per_second: f64) -> Option<Self> {
        if footprint == 0 || !bytes_per_second.is_finite() || bytes_per_second <= 0.0 {
            return None;
        }
        self.reused_read = Some((footprint, bytes_per_second));
        Some(self)
    }

    /// Prices cooperative critical paths by their serial resource sum.
    /// Rates must be half measured rates; the finite margin must be at least one.
    #[must_use]
    pub fn with_serial_cooperative_margin(mut self, margin: f64) -> Option<Self> {
        if !margin.is_finite() || margin < 1.0 {
            return None;
        }
        self.serial_cooperative_margin = Some(margin);
        Some(self)
    }

    /// Applies a positive host scheduling allowance to each physical launch.
    /// Raw device observations remain unchanged.
    #[must_use]
    pub fn with_launch_floor(mut self, floor: Duration) -> Option<Self> {
        if floor.is_zero() {
            return None;
        }
        for rate in &mut self.rates {
            rate.rates[6] = rate.rates[6].max(floor.as_secs_f64());
        }
        self.launch_floor = Some(floor);
        Some(self)
    }

    /// Returns the independently calibrated allowance for cold command-buffer startup.
    #[must_use]
    pub const fn command_buffer_reserve(&self) -> Duration {
        self.reserve
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
        let aggregate = match (access, self.reused_read) {
            (MemoryAccess::Reused { footprint }, Some((capacity, rate)))
                if footprint <= capacity =>
            {
                self.rates[0].predict_reused(total, launches.max(1), footprint, rate)?
            }
            _ => self.rates[0].predict(total, launches.max(1), access)?,
        };
        critical.read_bytes = critical
            .read_bytes
            .checked_sub(critical.metadata_read_bytes)?;
        let metadata = critical.metadata_read_bytes as f64 / self.dependent_read;
        let critical = self.rates[if cooperative { 2 } else { 1 }].predict_with_margin(
            critical,
            launches.max(1),
            access,
            if cooperative {
                self.serial_cooperative_margin
            } else {
                None
            },
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
        serde_json::json!({"kind":"admission calibration, not model throughput", "reused":self.reused,
            "rate_policy":"zero warmups; half steady rates per identical workload; twice worst launch/group latency and within-probe GPU spread",
            "command_buffer_reserve_seconds":self.reserve.as_secs_f64(),
            "host_launch_floor_seconds":self.launch_floor.map(|d| d.as_secs_f64()),
            "serial_cooperative_margin":self.serial_cooperative_margin,
            "reused_read": self.reused_read.map(|(bytes, rate)| serde_json::json!({"footprint_bytes":bytes,"bytes_per_second":rate})),
            "aggregate":self.rates[0].json(),"serial":self.rates[1].json(),
            "cooperative":self.rates[2].json(),"group_seconds":self.group_seconds, "group_reference_threads":self.group_threads,
            "dependent_read_bytes_per_second":self.dependent_read,
            "observations": self.observations.iter().map(|s| serde_json::json!({"name":s.name,
                "kernel":s.kernel,"units":s.units,"gpu_seconds":s.gpu.as_secs_f64()})).collect::<Vec<_>>()})
    }
}

impl WorkCalibration {
    /// Encodes calibration with the exact host-controlled source and device identity.
    #[must_use]
    pub fn cache_record(&self, identity: &str) -> String {
        serde_json::json!({"identity":identity,"calibration":self.json()}).to_string()
    }

    /// Revalidates cached observations and all rates, refusing a different identity.
    #[must_use]
    pub fn from_cache_record(record: &str, identity: &str) -> Option<Self> {
        let root: serde_json::Value = serde_json::from_str(record).ok()?;
        if root["identity"].as_str()? != identity {
            return None;
        }
        let value = &root["calibration"];
        let rates = |name| {
            let rate = &value[name];
            WorkRates::new(
                rate["read_bytes_per_second"].as_f64()?,
                rate["copy_bytes_per_second"].as_f64()?,
                rate["gather_bytes_per_second"].as_f64()?,
                rate["matrix_flops_per_second"].as_f64()?,
                rate["scalar_ops_per_second"].as_f64()?,
                rate["sfu_ops_per_second"].as_f64()?,
                rate["dependent_launch_seconds"].as_f64()?,
            )
        };
        let observations = value["observations"]
            .as_array()?
            .iter()
            .map(|s| {
                Some(WorkObservation {
                    name: s["name"].as_str()?.to_owned(),
                    kernel: s["kernel"].as_str()?.to_owned(),
                    units: s["units"].as_u64()?,
                    gpu: Duration::try_from_secs_f64(s["gpu_seconds"].as_f64()?).ok()?,
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let mut result = Self::new(
            [rates("aggregate")?, rates("serial")?, rates("cooperative")?],
            value["group_seconds"].as_f64()?,
            value["group_reference_threads"].as_u64()?,
            value["dependent_read_bytes_per_second"].as_f64()?,
            observations,
            Duration::try_from_secs_f64(value["command_buffer_reserve_seconds"].as_f64()?).ok()?,
        )?;
        if !value["reused_read"].is_null() {
            result = result.with_reused_read(
                value["reused_read"]["footprint_bytes"].as_u64()?,
                value["reused_read"]["bytes_per_second"].as_f64()?,
            )?;
        }
        if !value["host_launch_floor_seconds"].is_null() {
            result = result.with_launch_floor(
                Duration::try_from_secs_f64(value["host_launch_floor_seconds"].as_f64()?).ok()?,
            )?;
        }
        if !value["serial_cooperative_margin"].is_null() {
            result = result
                .with_serial_cooperative_margin(value["serial_cooperative_margin"].as_f64()?)?;
        }
        result.reused = true;
        Some(result)
    }
}
