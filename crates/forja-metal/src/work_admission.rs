use crate::{CalibrationSample, MetalBackend};
use forja_core::{
    BackendError, Dispatch, MemoryAccess, Op, StateOp, WorkCalibration, WorkObservation, WorkRates,
    program::ProgramKind,
};
use objc2_metal::MTLDevice;
use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

thread_local! { static CALIBRATING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
pub(super) struct CalibrationScope(bool);
impl CalibrationScope {
    pub(super) fn enter() -> Self {
        Self(CALIBRATING.replace(true))
    }
}
impl Drop for CalibrationScope {
    fn drop(&mut self) {
        CALIBRATING.set(self.0);
    }
}
pub(super) fn calibrating() -> bool {
    CALIBRATING.get()
}

static CALIBRATION: OnceLock<Result<(u64, Arc<WorkCalibration>), BackendError>> = OnceLock::new();

pub(super) fn calibration(backend: &MetalBackend) -> Result<Arc<WorkCalibration>, BackendError> {
    let (device, rates) = CALIBRATION
        .get_or_init(|| {
            let probe = MetalBackend::new()?;
            let mut samples = probe.calibrate_memory(1, 6)?;
            samples.extend(probe.calibrate_compute(1, 6)?);
            samples.extend(probe.calibrate_dispatch(1, 6)?);
            samples.extend(probe.calibrate_work(1, 6)?);
            samples.extend(probe.calibrate_compute_group(6)?);
            samples.extend(probe.calibrate_work_group(6)?);
            Ok((probe.device.registryID(), Arc::new(from_samples(&samples)?)))
        })
        .as_ref()
        .map_err(|e| *e)?;
    if *device != backend.device.registryID() {
        return Err(BackendError::UnsupportedOperation);
    }
    Ok(Arc::clone(rates))
}

#[allow(clippy::cast_precision_loss)]
fn from_samples(samples: &[CalibrationSample]) -> Result<WorkCalibration, BackendError> {
    let rate = |names: &[&str]| -> Result<f64, BackendError> {
        let mut minimum = f64::INFINITY;
        for name in names {
            let observations = samples
                .iter()
                .filter(|s| !s.screening && s.name == *name)
                .collect::<Vec<_>>();
            if observations.is_empty() {
                return Err(BackendError::InvalidInput);
            }
            for s in observations {
                let value = s.units as f64 / s.gpu.as_secs_f64();
                if !value.is_finite() || value <= 0.0 {
                    return Err(BackendError::InvalidInput);
                }
                minimum = minimum.min(value);
            }
        }
        Ok(minimum * 0.5)
    };
    let read = rate(&["streaming_read"])?;
    let copy = rate(&["streaming_copy"])?;
    let gather = rate(&["calibration_gather"])?;
    let matrix = rate(&[
        "calibration_mma_f32",
        "calibration_mma_f16",
        "calibration_mma_bf16",
    ])?;
    let launch = 1.0 / rate(&["dependent_dispatch"])?;
    let serial_read = rate(&["calibration_read_serial"])?;
    let dependent = rate(&["calibration_dependent_read"])?;
    let serial = rate(&["calibration_scalar_serial"])?;
    let rates = [
        WorkRates::new(
            read,
            copy,
            gather,
            matrix,
            rate(&["calibration_scalar_parallel"])?,
            rate(&["calibration_sfu"])?,
            launch,
        ),
        WorkRates::new(
            serial_read,
            serial_read,
            dependent,
            serial,
            serial,
            rate(&["calibration_sfu_serial"])?,
            launch,
        ),
        WorkRates::new(
            rate(&["calibration_read_wide_group"])?,
            rate(&["calibration_read_wide_group"])?,
            rate(&["calibration_gather_group"])?,
            rate(&["calibration_mma_group"])?,
            rate(&["calibration_scalar_wide_group"])?,
            rate(&["calibration_sfu_wide_group"])?,
            launch,
        ),
    ];
    let rates = [
        rates[0].ok_or(BackendError::InvalidInput)?,
        rates[1].ok_or(BackendError::InvalidInput)?,
        rates[2].ok_or(BackendError::InvalidInput)?,
    ];
    WorkCalibration::new(
        rates,
        1.0 / rate(&["calibration_group_dispatch"])?,
        u64::from(crate::calibration::THREADS),
        dependent,
        samples
            .iter()
            .filter(|s| !s.screening)
            .map(|s| WorkObservation {
                name: format!(
                    "{} {:?} chains={}",
                    s.name, s.arithmetic, s.independent_chains
                ),
                kernel: s.kernel.to_owned(),
                units: s.units,
                gpu: s.gpu,
            })
            .collect(),
    )
    .ok_or(BackendError::InvalidInput)
}

pub(super) fn trusted_probe(d: &Dispatch) -> bool {
    d.backend_data::<crate::calibration::Probe>().is_some()
        || d.backend_data::<crate::work_calibration::WorkProbe>()
            .is_some()
}

pub(super) fn predict(
    rates: &WorkCalibration,
    d: &Dispatch,
    launches: u64,
    groups: u64,
    threads: u64,
    schedule: Option<crate::encoding::AttentionSchedule>,
) -> Result<Duration, BackendError> {
    let (mut total, mut critical) = if let Some(schedule) = schedule {
        crate::profile_cost::scheduled::attention_cost(d, schedule)
            .ok_or(BackendError::InvalidInput)?
    } else {
        (
            crate::profile_cost::scheduled::of(d).ok_or(BackendError::InvalidInput)?,
            crate::profile_cost::scheduled::critical_path(d).ok_or(BackendError::InvalidInput)?,
        )
    };
    let attention = schedule.is_some();
    if matches!(
        schedule,
        Some(crate::encoding::AttentionSchedule::Vector { .. })
    ) {
        for cost in [&mut total, &mut critical] {
            cost.scalar_ops = cost
                .scalar_ops
                .checked_add(cost.flops)
                .ok_or(BackendError::InvalidInput)?;
            cost.flops = 0;
        }
    }
    let sparse_scalar =
        crate::profile_cost::scheduled::scalar_matrix_groups(d).is_some_and(|groups| groups < 32);
    let cooperative = !sparse_scalar
        && (attention
            || matches!(
                d.op(),
                Op::RmsNorm { .. }
                    | Op::Sample { .. }
                    | Op::SampleMasked { .. }
                    | Op::Matmul
                    | Op::QuantMatmul { .. }
                    | Op::State(StateOp::DeltaRule(_) | StateOp::DeltaRuleBlock(_))
            )
            || d.bound_program()
                .is_some_and(|p| p.program().program().kind == ProgramKind::Row));
    let scattered = matches!(d.op(), Op::GatherMatmul | Op::GatherQuantMatmul { .. })
        || d.inputs().iter().any(|t| {
            t.layout()
                .strides()
                .last()
                .is_some_and(|&stride| stride != 1)
        });
    rates
        .predict(
            total,
            critical,
            cooperative,
            launches,
            (groups, threads),
            if scattered {
                MemoryAccess::Scattered
            } else {
                MemoryAccess::Coalesced
            },
        )
        .ok_or(BackendError::InvalidInput)
}

impl MetalBackend {
    /// Prepares device-specific admission calibration before loading an engine.
    /// # Errors
    /// Refuses unsupported devices or failed calibration.
    pub fn prepare_work_estimation(&self) -> Result<(), BackendError> {
        calibration(self).map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn calibration_refuses_missing_zero_or_unmeasured_rates() {
        assert!(from_samples(&[]).is_err());
        let samples = vec![CalibrationSample {
            name: "streaming_read",
            kernel: "stream",
            arithmetic: None,
            independent_chains: 0,
            screening: false,
            units: 0,
            wall: Duration::ZERO,
            gpu: Duration::ZERO,
        }];
        assert!(from_samples(&samples).is_err());
    }
}
