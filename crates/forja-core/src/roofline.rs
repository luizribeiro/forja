//! Logical roofline bounds using measured, device-independent resource ceilings.

mod estimate;
pub use estimate::Estimate;

use crate::KernelCost;

#[derive(Clone, Copy, Debug)]
/// Validated measured resource ceilings.
pub struct Ceilings {
    read_bytes_per_second: f64,
    copy_bytes_per_second: f64,
    flops_per_second: f64,
    dispatch_seconds: f64,
    rates_low: [f64; 3],
    rates_high: [f64; 3],
    dispatch_interval: [f64; 2],
}

#[derive(Clone, Copy, Debug)]
/// Competing resource bounds, all expressed in seconds.
pub struct ResourceTimes {
    memory: f64,
    compute: f64,
    dispatch: f64,
}

impl Ceilings {
    /// Constructs ceilings only from finite positive rates and dispatch latency.
    #[must_use]
    pub fn new(
        read_bytes_per_second: f64,
        copy_bytes_per_second: f64,
        flops_per_second: f64,
        dispatch_seconds: f64,
    ) -> Option<Self> {
        Some(Self::with_intervals(
            Estimate::new(read_bytes_per_second, [read_bytes_per_second; 2])?,
            Estimate::new(copy_bytes_per_second, [copy_bytes_per_second; 2])?,
            Estimate::new(flops_per_second, [flops_per_second; 2])?,
            Estimate::new(dispatch_seconds, [dispatch_seconds; 2])?,
        ))
    }

    /// Retains measured confidence intervals for rates and dependent-dispatch latency.
    #[must_use]
    pub fn with_intervals(
        read: Estimate,
        copy: Estimate,
        compute: Estimate,
        dispatch: Estimate,
    ) -> Self {
        Self {
            read_bytes_per_second: read.median(),
            copy_bytes_per_second: copy.median(),
            flops_per_second: compute.median(),
            dispatch_seconds: dispatch.median(),
            rates_low: [read.lower(), copy.lower(), compute.lower()],
            rates_high: [read.upper(), copy.upper(), compute.upper()],
            dispatch_interval: [dispatch.lower(), dispatch.upper()],
        }
    }

    /// Uses the fastest ends of the measured rate and latency intervals.
    #[must_use]
    pub const fn optimistic(self) -> Self {
        Self {
            read_bytes_per_second: self.rates_high[0],
            copy_bytes_per_second: self.rates_high[1],
            flops_per_second: self.rates_high[2],
            dispatch_seconds: self.dispatch_interval[0],
            ..self
        }
    }

    /// Uses the slowest ends of the measured rate and latency intervals.
    #[must_use]
    pub const fn pessimistic(self) -> Self {
        Self {
            read_bytes_per_second: self.rates_low[0],
            copy_bytes_per_second: self.rates_low[1],
            flops_per_second: self.rates_low[2],
            dispatch_seconds: self.dispatch_interval[1],
            ..self
        }
    }
    /// Returns the streaming read ceiling in bytes per second.
    #[must_use]
    pub const fn read_bandwidth(self) -> f64 {
        self.read_bytes_per_second
    }
    /// Returns the combined streaming read/write ceiling in bytes per second.
    #[must_use]
    pub const fn copy_bandwidth(self) -> f64 {
        self.copy_bytes_per_second
    }
    /// Returns the matrix-product ceiling in FLOPs per second.
    #[must_use]
    pub const fn compute_rate(self) -> f64 {
        self.flops_per_second
    }
    /// Returns the dependent-dispatch floor in seconds.
    #[must_use]
    pub const fn dispatch_floor(self) -> f64 {
        self.dispatch_seconds
    }
}

impl ResourceTimes {
    /// Returns the dominant resource bound.
    #[must_use]
    pub fn seconds(self) -> f64 {
        self.memory.max(self.compute).max(self.dispatch)
    }

    /// Names the dominant calibrated resource.
    #[must_use]
    pub fn bound(self) -> &'static str {
        if self.dispatch >= self.memory.max(self.compute) {
            "dispatch-floor"
        } else if self.memory >= self.compute {
            "memory"
        } else {
            "compute"
        }
    }

    /// Returns a target bound, or `None` for an invalid fraction or overflow.
    #[must_use]
    pub fn target_seconds(self, utilization: f64) -> Option<f64> {
        if !(utilization.is_finite() && 0.0 < utilization && utilization <= 1.0) {
            return None;
        }
        let target = (self.memory.max(self.compute) / utilization).max(self.dispatch);
        target.is_finite().then_some(target)
    }
}

/// Counts operand traffic at one end of a declared expert-reuse interval.
/// Returns `None` on overflow or inconsistent byte counts.
#[must_use]
pub fn traffic(cost: KernelCost, maximum_reuse: bool) -> Option<u64> {
    let weights = if cost.maximum_weight_bytes == 0 {
        cost.weight_bytes
    } else if maximum_reuse {
        cost.minimum_weight_bytes
    } else {
        cost.maximum_weight_bytes
    };
    cost.read_bytes
        .checked_sub(cost.weight_bytes)?
        .checked_add(weights)?
        .checked_add(cost.write_bytes)
}

#[allow(clippy::cast_precision_loss)]
/// Computes competing memory, matrix-product and dependent-dispatch bounds.
/// Returns `None` on arithmetic overflow or inconsistent byte counts.
#[must_use]
pub fn limits(cost: KernelCost, dispatches: u64, ceilings: Ceilings) -> Option<ResourceTimes> {
    let bytes = traffic(cost, true)?;
    let read_bytes = bytes.checked_sub(cost.write_bytes)?;
    let times = ResourceTimes {
        memory: (read_bytes as f64 / ceilings.read_bytes_per_second)
            .max(bytes as f64 / ceilings.copy_bytes_per_second),
        compute: cost.flops as f64 / ceilings.flops_per_second,
        dispatch: dispatches as f64 * ceilings.dispatch_seconds,
    };
    [times.memory, times.compute, times.dispatch]
        .iter()
        .all(|value| value.is_finite())
        .then_some(times)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn ceilings() -> Ceilings {
        Ceilings::new(100.0, 200.0, 1000.0, 0.01).unwrap()
    }

    #[test]
    fn calibration_intervals_bound_rates_and_invert_dispatch_latency() {
        let measured = Ceilings::with_intervals(
            Estimate::new(100.0, [80.0, 125.0]).unwrap(),
            Estimate::new(200.0, [160.0, 250.0]).unwrap(),
            Estimate::new(1000.0, [800.0, 1250.0]).unwrap(),
            Estimate::new(0.01, [0.008, 0.0125]).unwrap(),
        );
        for cost in [
            KernelCost {
                read_bytes: 100,
                ..KernelCost::default()
            },
            KernelCost {
                flops: 1000,
                ..KernelCost::default()
            },
        ] {
            assert!(
                (limits(cost, 1, measured.optimistic()).unwrap().seconds() - 0.8).abs()
                    < f64::EPSILON
            );
            assert!(
                (limits(cost, 1, measured.pessimistic()).unwrap().seconds() - 1.25).abs()
                    < f64::EPSILON
            );
        }
        assert!(
            (limits(KernelCost::default(), 100, measured.optimistic())
                .unwrap()
                .seconds()
                - 0.8)
                .abs()
                < f64::EPSILON
        );
    }

    #[test]
    fn resource_bounds_compete_instead_of_being_added() {
        let cost = KernelCost {
            read_bytes: 100,
            write_bytes: 100,
            flops: 500,
            ..KernelCost::default()
        };
        let limit = limits(cost, 1, ceilings()).unwrap();
        assert_eq!(limit.bound(), "memory");
        assert!((limit.seconds() - 1.0).abs() < f64::EPSILON);
        assert!((limit.target_seconds(0.8).unwrap() - 1.25).abs() < f64::EPSILON);
        let limit = limits(
            KernelCost {
                flops: 2000,
                ..KernelCost::default()
            },
            1,
            ceilings(),
        )
        .unwrap();
        assert_eq!(limit.bound(), "compute");
        let limit = limits(KernelCost::default(), 7, ceilings()).unwrap();
        assert_eq!(limit.bound(), "dispatch-floor");
        assert!((limit.seconds() - 0.07).abs() < f64::EPSILON);
    }

    #[test]
    fn expert_reuse_uncertainty_remains_explicit() {
        let cost = KernelCost {
            read_bytes: 1000,
            weight_bytes: 900,
            minimum_weight_bytes: 10,
            maximum_weight_bytes: 100,
            ..KernelCost::default()
        };
        assert_eq!(traffic(cost, true), Some(110));
        assert_eq!(traffic(cost, false), Some(200));
        assert_eq!(
            traffic(
                KernelCost {
                    read_bytes: u64::MAX,
                    write_bytes: 1,
                    ..KernelCost::default()
                },
                true
            ),
            None
        );
        assert_eq!(
            traffic(
                KernelCost {
                    read_bytes: 0,
                    weight_bytes: 1,
                    ..KernelCost::default()
                },
                true
            ),
            None
        );
    }
    #[test]
    fn ceilings_refuse_nonfinite_or_nonpositive_values() {
        assert!(Ceilings::new(0.0, 1.0, 1.0, 1.0).is_none());
        assert!(Ceilings::new(1.0, f64::INFINITY, 1.0, 1.0).is_none());
        assert!(
            limits(KernelCost::default(), 1, ceilings())
                .unwrap()
                .target_seconds(0.0)
                .is_none()
        );
    }
}
