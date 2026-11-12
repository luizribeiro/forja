mod cache;
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

#[cfg(test)]
thread_local! { static TEST_CALIBRATION: std::cell::RefCell<Option<Arc<WorkCalibration>>> = const { std::cell::RefCell::new(None) }; }
#[cfg(test)]
pub(crate) struct TestCalibration(Option<Arc<WorkCalibration>>);
#[cfg(test)]
impl TestCalibration {
    pub(crate) fn fast() -> Self {
        let rates = WorkRates::new(1e15, 1e15, 1e15, 1e15, 1e15, 1e15, 1e-9).unwrap();
        let rates = WorkCalibration::new(
            [rates; 3],
            1e-15,
            128,
            1e15,
            vec![WorkObservation {
                name: "mock".into(),
                kernel: "mock".into(),
                units: 1,
                gpu: Duration::from_nanos(1),
            }],
            Duration::ZERO,
        )
        .unwrap();
        Self(TEST_CALIBRATION.replace(Some(Arc::new(rates))))
    }
}
#[cfg(test)]
impl Drop for TestCalibration {
    fn drop(&mut self) {
        TEST_CALIBRATION.set(self.0.take());
    }
}

static CALIBRATION: OnceLock<Result<(u64, Arc<WorkCalibration>), BackendError>> = OnceLock::new();

pub(super) fn calibration(backend: &MetalBackend) -> Result<Arc<WorkCalibration>, BackendError> {
    #[cfg(test)]
    if let Some(rates) = TEST_CALIBRATION.with_borrow(Clone::clone) {
        return Ok(rates);
    }

    let (device, rates) = CALIBRATION
        .get_or_init(|| {
            let _scope = CalibrationScope::enter();
            let cache = cache::Cache::new(backend);
            if let Some(rates) = cache.as_ref().and_then(cache::Cache::read) {
                return Ok((backend.device.registryID(), Arc::new(rates)));
            }
            let probe = MetalBackend::new()?;
            let mut samples = probe.calibrate_memory(0, 6)?;
            samples.extend(probe.calibrate_compute(0, 6)?);
            samples.extend(probe.calibrate_dispatch(0, 6)?);
            samples.extend(probe.calibrate_work(0, 6)?);
            samples.extend(probe.calibrate_compute_group(6)?);
            samples.extend(probe.calibrate_work_group(6)?);
            let rates = from_samples(&samples)?;
            if let Some(cache) = cache {
                cache.write(&rates);
            }
            Ok((probe.device.registryID(), Arc::new(rates)))
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
    let rate = |names: &[&str]| sample_rate(samples, names, true);
    let latency = |names: &[&str]| sample_rate(samples, names, false);
    let read = rate(&["streaming_read"])?;
    let copy = rate(&["streaming_copy"])?;
    let gather = rate(&["calibration_gather"])?;
    let matrix = rate(&[
        "calibration_mma_f32",
        "calibration_mma_f16",
        "calibration_mma_bf16",
    ])?;
    let launch = 1.0 / latency(&["dependent_dispatch"])?;
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
    let reused_read = rate(&["calibration_reused_read"])?;
    WorkCalibration::new(
        rates,
        1.0 / latency(&["calibration_group_dispatch"])?,
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
        startup_reserve(samples)?,
    )
    .and_then(|rates| {
        rates.with_reused_read(crate::work_calibration::REUSED_READ_BYTES, reused_read)
    })
    .ok_or(BackendError::InvalidInput)
}

fn same_work(a: &CalibrationSample, b: &CalibrationSample) -> bool {
    a.name == b.name
        && a.kernel == b.kernel
        && a.units == b.units
        && a.arithmetic == b.arithmetic
        && a.independent_chains == b.independent_chains
}

#[allow(clippy::cast_precision_loss)]
fn sample_rate(
    samples: &[CalibrationSample],
    names: &[&str],
    steady: bool,
) -> Result<f64, BackendError> {
    let mut minimum = f64::INFINITY;
    for name in names {
        let mut found = false;
        for sample in samples.iter().filter(|s| !s.screening && s.name == *name) {
            found = true;
            if sample.units == 0 || sample.gpu.is_zero() {
                return Err(BackendError::InvalidInput);
            }
            let time = if steady {
                samples
                    .iter()
                    .filter(|s| !s.screening && same_work(s, sample))
                    .map(|s| s.gpu)
                    .min()
                    .ok_or(BackendError::InvalidInput)?
            } else {
                sample.gpu
            };
            minimum = minimum.min(sample.units as f64 / time.as_secs_f64());
        }
        if !found {
            return Err(BackendError::InvalidInput);
        }
    }
    let rate = minimum * 0.5;
    if rate.is_finite() && rate > 0.0 {
        Ok(rate)
    } else {
        Err(BackendError::InvalidInput)
    }
}

fn startup_reserve(samples: &[CalibrationSample]) -> Result<Duration, BackendError> {
    samples
        .iter()
        .filter(|s| !s.screening)
        .map(|sample| {
            let steady = samples
                .iter()
                .filter(|s| !s.screening && same_work(s, sample))
                .map(|s| s.gpu)
                .min()
                .unwrap_or(sample.gpu);
            sample.gpu.saturating_sub(steady)
        })
        .max()
        .unwrap_or_default()
        .checked_mul(2)
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
            } else if let Some(footprint) = tiled_input_footprint(d) {
                MemoryAccess::Reused { footprint }
            } else {
                MemoryAccess::Coalesced
            },
        )
        .ok_or(BackendError::InvalidInput)
}

fn tiled_input_footprint(dispatch: &Dispatch) -> Option<u64> {
    if !matches!(
        crate::encoding::selected_variant(dispatch),
        Some(
            crate::variants::MetalVariant::QuantMatmulTiled
                | crate::variants::MetalVariant::QuantMatmulTiledK64
        )
    ) {
        return None;
    }
    dispatch.inputs().iter().try_fold(0_u64, |bytes, tensor| {
        let layout = tensor.layout();
        if !layout.is_contiguous() {
            return None;
        }
        let span = layout.byte_span();
        bytes.checked_add(span.end.checked_sub(span.start)?)
    })
}

pub(super) type WorkFaults = forja_core::GpuWorkTracker<Arc<[usize]>>;
pub(super) type Backstop = forja_core::GpuWorkTicket<Arc<[usize]>>;

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
    fn tile_reread_pricing_requires_a_validated_coalesced_schedule() {
        use forja_core::{BackendDispatchData, CommandList, DType, DeviceCapability};
        let cpu = forja_cpu::CpuBackend::new();
        let x = cpu.alloc(DType::BF16, &[33, 128]).unwrap();
        let w = cpu.alloc(DType::U32, &[128, 16]).unwrap();
        let scale = cpu.alloc(DType::BF16, &[128, 2]).unwrap();
        let out = cpu.alloc(DType::BF16, &[33, 128]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::QuantMatmul {
                    bits: 4,
                    group_size: 64,
                },
                &[&x, &w, &scale, &scale],
                &out,
            )
            .unwrap();
        let dispatch = commands.last_dispatch().unwrap();
        assert!(tiled_input_footprint(dispatch).is_none());
        let variant = crate::variants::resolve(
            dispatch,
            "quant-matmul.tiled-k64",
            &[DeviceCapability::Metal4],
        )
        .unwrap();
        let dispatch = dispatch
            .clone()
            .with_backend_data(BackendDispatchData::new(variant));
        assert_eq!(
            tiled_input_footprint(&dispatch),
            Some(33 * 128 * 2 + 128 * 16 * 4 + 128 * 2 * 2 * 2)
        );
    }

    #[test]
    fn startup_reserve_compares_only_identical_probe_workloads() {
        let sample = CalibrationSample {
            name: "streaming_read",
            kernel: "read",
            arithmetic: None,
            independent_chains: 0,
            screening: false,
            units: 1024,
            wall: Duration::from_millis(7),
            gpu: Duration::from_millis(7),
        };
        let mut samples = vec![sample.clone()];
        samples.push(CalibrationSample {
            gpu: Duration::from_millis(2),
            ..sample.clone()
        });
        samples.push(CalibrationSample {
            units: 4097,
            gpu: Duration::from_secs(1),
            ..sample.clone()
        });
        samples.push(CalibrationSample {
            screening: true,
            gpu: Duration::from_secs(2),
            ..sample
        });
        assert_eq!(startup_reserve(&samples), Ok(Duration::from_millis(10)));
    }

    #[test]
    fn sustained_rate_plus_startup_bounds_every_raw_observation() {
        let samples = [2_u64, 7, 3].map(|ms| CalibrationSample {
            name: "streaming_read",
            kernel: "read",
            arithmetic: None,
            independent_chains: 0,
            screening: false,
            units: 1024,
            wall: Duration::from_millis(ms),
            gpu: Duration::from_millis(ms),
        });
        let steady = sample_rate(&samples, &["streaming_read"], true).unwrap();
        let latency = sample_rate(&samples, &["streaming_read"], false).unwrap();
        assert!(steady > latency);
        let forecast =
            Duration::from_secs_f64(1024.0 / steady) + startup_reserve(&samples).unwrap();
        for sample in samples {
            assert!(forecast >= sample.gpu * 2);
        }
    }

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
