use crate::MetalBackend;
use forja_core::{
    Backend, BackendDispatchData, BackendError, CommandList, DType, Op, Submission, Tensor,
};
use objc2_metal::MTLComputePipelineState;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub(crate) enum Probe {
    Mma(DType, Chains),
    NativeBf16(Chains),
    Threadgroup,
}

#[derive(Clone, Copy)]
pub(crate) enum Chains {
    Four,
}

impl Chains {
    const fn count(self) -> u32 {
        match self {
            Self::Four => 4,
        }
    }
}

impl Probe {
    pub(crate) const fn kernel(self) -> &'static str {
        match self {
            Self::Mma(DType::F16, _) => "calibration_mma_f16",
            Self::Mma(DType::BF16, _) => "calibration_mma_bf16",
            Self::Mma(_, _) => "calibration_mma_f32",
            Self::NativeBf16(_) => "calibration_mma_bf16_native",
            Self::Threadgroup => "calibration_threadgroup",
        }
    }
    const fn label(self) -> &'static str {
        match self {
            Self::NativeBf16(_) => "calibration_mma_bf16",
            Self::Mma(_, _) | Self::Threadgroup => self.kernel(),
        }
    }
    const fn independent_chains(self) -> u32 {
        match self {
            Self::Mma(_, chains) | Self::NativeBf16(chains) => chains.count(),
            Self::Threadgroup => 0,
        }
    }
    pub(crate) const fn constants(self) -> [(u32, u32); 2] {
        [(220, ITERATIONS), (221, self.independent_chains())]
    }
}

const GROUPS: u32 = 2048;
pub(crate) const THREADS: u32 = 128;
pub(crate) const ITERATIONS: u32 = 4096;

/// One unprofiled device calibration observation, never a model benchmark.
#[derive(Clone, Debug)]
pub struct CalibrationSample {
    /// Probe identity and units (bytes, FLOPs, or dispatches).
    pub name: &'static str,
    /// Probe implementation identifier; MMA records the exact compiler function.
    pub kernel: &'static str,
    /// Arithmetic type actually executed, when this is an MMA probe.
    pub arithmetic: Option<DType>,
    /// Independent accumulator chains; zero for probes without MMA.
    pub independent_chains: u32,
    /// Whether this observation selected a configuration rather than estimating its ceiling.
    pub screening: bool,
    /// Work units executed in the timed command buffer.
    pub units: u64,
    /// Elapsed host time including submission and completion.
    pub wall: Duration,
    /// Command-buffer GPU time, without per-dispatch timestamps.
    pub gpu: Duration,
}

#[derive(Clone, Copy)]
struct PreparedProbe {
    probe: Probe,
    arithmetic: Option<DType>,
    simdgroups: u32,
}

impl MetalBackend {
    /// Measures prepared MMA configurations and threadgroup reads.
    ///
    /// BF16 compiler refusal falls back to explicitly labelled f32 promotion.
    ///
    /// # Errors
    /// Returns an error for invalid counts or allocation, compilation, execution or timing failure.
    pub fn calibrate_compute(
        &self,
        warmups: u32,
        repetitions: u32,
    ) -> Result<Vec<CalibrationSample>, BackendError> {
        if repetitions == 0 {
            return Err(BackendError::InvalidInput);
        }
        let _calibration = self
            .calibration_lock
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let input = self.alloc(DType::F32, &[GROUPS * THREADS])?;
        let output = self.alloc(DType::F32, &[GROUPS * THREADS])?;
        self.write(
            &input,
            &vec![
                1.0_f32.to_le_bytes();
                usize::try_from(GROUPS * THREADS).map_err(|_| BackendError::InvalidInput)?
            ]
            .concat(),
        )?;
        let result = (|| {
            let mut samples = Vec::new();
            for probe in [
                Probe::Mma(DType::F32, Chains::Four),
                Probe::Mma(DType::F16, Chains::Four),
                Probe::Mma(DType::BF16, Chains::Four),
                Probe::Threadgroup,
            ] {
                let prepared = self.prepare_calibration_probe(probe)?;
                samples.extend(self.compute_probe_samples(
                    prepared,
                    [&input, &output],
                    warmups,
                    repetitions,
                    false,
                )?);
            }
            Ok(samples)
        })();
        self.release(&input)?;
        self.release(&output)?;
        result
    }

    fn prepare_calibration_probe(&self, probe: Probe) -> Result<PreparedProbe, BackendError> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let probe = if let Probe::Mma(DType::BF16, chains) = probe {
            let native = Probe::NativeBf16(chains);
            let source = format!(
                "#define FORJA_CALIBRATION_NATIVE_BF16 1\n#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n{}",
                include_str!("calibration.metal")
            );
            if pipelines.try_calibration_shader(&source, native.kernel(), &native.constants())? {
                native
            } else {
                probe
            }
        } else {
            probe
        };
        let width = u32::try_from(
            pipelines
                .get(probe.kernel(), &probe.constants())?
                .threadExecutionWidth(),
        )
        .map_err(|_| BackendError::InvalidInput)?;
        let simdgroups = THREADS
            .checked_div(width)
            .filter(|_| THREADS.is_multiple_of(width))
            .ok_or(BackendError::InvalidInput)?;
        let arithmetic = match probe {
            Probe::Mma(DType::BF16, _) => Some(DType::F32),
            Probe::NativeBf16(_) => Some(DType::BF16),
            Probe::Mma(dtype, _) => Some(dtype),
            Probe::Threadgroup => None,
        };
        Ok(PreparedProbe {
            probe,
            arithmetic,
            simdgroups,
        })
    }

    fn compute_probe_samples(
        &self,
        prepared: PreparedProbe,
        tensors: [&Tensor; 2],
        warmups: u32,
        repetitions: u32,
        screening: bool,
    ) -> Result<Vec<CalibrationSample>, BackendError> {
        let units = match prepared.probe {
            Probe::Mma(_, chains) | Probe::NativeBf16(chains) => [
                u64::from(prepared.simdgroups),
                u64::from(chains.count()),
                2 * 8 * 8 * 8,
            ],
            Probe::Threadgroup => [u64::from(THREADS), 16, 1],
        }
        .into_iter()
        .chain([u64::from(GROUPS), u64::from(ITERATIONS)])
        .try_fold(1_u64, u64::checked_mul)
        .ok_or(BackendError::InvalidInput)?;
        let [input, output] = tensors;
        let mut samples =
            self.calibration_samples(prepared.probe.label(), units, warmups, repetitions, || {
                let mut commands = CommandList::new();
                commands
                    .dispatch_many_with_backend_data(
                        Op::Copy,
                        &[input],
                        &[output],
                        BackendDispatchData::new(prepared.probe),
                    )
                    .map_err(|_| BackendError::InvalidInput)?;
                Ok(commands)
            })?;
        for sample in &mut samples {
            sample.kernel = prepared.probe.kernel();
            sample.arithmetic = prepared.arithmetic;
            sample.independent_chains = prepared.probe.independent_chains();
            sample.screening = screening;
        }
        let values = self.read(output)?;
        let (minimum, maximum) = match prepared.probe {
            Probe::Mma(_, chains) | Probe::NativeBf16(chains) => {
                let chains = f64::from(chains.count());
                let initial = chains * (chains + 1.0);
                let work = 2.0 * chains * 8.0 * f64::from(ITERATIONS) * 0.001 * 0.002;
                (initial + work * 0.1, initial + work * 4.0)
            }
            Probe::Threadgroup => (f64::from(ITERATIONS), f64::from(ITERATIONS) * 8.0),
        };
        if values.as_chunks::<4>().0.iter().copied().any(|bytes| {
            let value = f64::from(f32::from_le_bytes(bytes));
            !value.is_finite() || value < minimum || value > maximum
        }) {
            return Err(BackendError::ExecutionFailed);
        }
        Ok(samples)
    }
}

impl MetalBackend {
    pub(super) fn calibration_samples(
        &self,
        name: &'static str,
        units: u64,
        warmups: u32,
        repetitions: u32,
        mut commands: impl FnMut() -> Result<CommandList, BackendError>,
    ) -> Result<Vec<CalibrationSample>, BackendError> {
        if repetitions == 0 {
            return Err(BackendError::InvalidInput);
        }
        let count = warmups
            .checked_add(repetitions)
            .ok_or(BackendError::InvalidInput)?;
        let mut samples = Vec::new();
        for index in 0..count {
            let commands = commands()?;
            let started = Instant::now();
            let submission = self.submit(commands)?;
            submission.wait()?;
            let wall = started.elapsed();
            let gpu = submission
                .gpu_time()
                .filter(|time| *time > Duration::ZERO)
                .ok_or(BackendError::ExecutionFailed)?;
            if index >= warmups {
                samples.push(CalibrationSample {
                    arithmetic: None,
                    independent_chains: 0,
                    screening: false,
                    name,
                    kernel: name,
                    units,
                    wall,
                    gpu,
                });
            }
        }
        Ok(samples)
    }

    /// Measures the unprofiled floor of a dependent alternating copy chain.
    ///
    /// # Errors
    /// Returns an error on invalid counts or allocation, execution, timing or release failure.
    pub fn calibrate_dispatch(
        &self,
        warmups: u32,
        repetitions: u32,
    ) -> Result<Vec<CalibrationSample>, BackendError> {
        let first = self.alloc(DType::F32, &[1])?;
        let second = self.alloc(DType::F32, &[1])?;
        let result =
            self.calibration_samples("dependent_dispatch", 512, warmups, repetitions, || {
                let mut commands = CommandList::new();
                for index in 0..512 {
                    let (input, output) = if index % 2 == 0 {
                        (&first, &second)
                    } else {
                        (&second, &first)
                    };
                    commands
                        .dispatch(Op::Copy, &[input], output)
                        .map_err(|_| BackendError::InvalidInput)?;
                }
                Ok(commands)
            });
        self.release(&first)?;
        self.release(&second)?;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_bf16_identifier_selects_an_explicit_bfloat_mma_kernel() {
        let native = Probe::NativeBf16(Chains::Four);
        let promoted = Probe::Mma(DType::BF16, Chains::Four);
        assert_eq!(native.kernel(), "calibration_mma_bf16_native");
        assert_ne!(native.kernel(), promoted.kernel());
        assert_eq!(native.label(), promoted.label());
        assert!(
            include_str!("calibration.metal")
                .contains("calibration_mma_kernel(calibration_mma_bf16_native, bfloat, bfloat)")
        );
    }

    #[test]
    fn metal_calibration_executes_observable_mma_and_threadgroup_work() {
        let backend = MetalBackend::new().unwrap();
        let samples = backend.calibrate_compute(0, 1).unwrap();
        assert_eq!(samples.iter().filter(|sample| sample.screening).count(), 0);
        let measured = samples
            .iter()
            .filter(|sample| !sample.screening)
            .collect::<Vec<_>>();
        assert_eq!(measured.len(), 4);
        assert_eq!(measured[0].arithmetic, Some(DType::F32));
        assert_eq!(measured[1].arithmetic, Some(DType::F16));
        assert!(matches!(
            measured[2].arithmetic,
            Some(DType::BF16 | DType::F32)
        ));
        assert_eq!(measured[2].name, "calibration_mma_bf16");
        assert_eq!(
            measured[2].kernel,
            if measured[2].arithmetic == Some(DType::BF16) {
                "calibration_mma_bf16_native"
            } else {
                "calibration_mma_bf16"
            }
        );
        assert_eq!(measured[3].arithmetic, None);
        assert!(
            measured[..3]
                .iter()
                .all(|sample| [4, 8, 16].contains(&sample.independent_chains))
        );
        assert!(samples.iter().all(|sample| sample.gpu > Duration::ZERO
            && sample.wall > Duration::ZERO
            && sample.units > 0));
        assert!(backend.calibrate_compute(0, 0).is_err());
        let chain = backend.calibrate_dispatch(1, 2).unwrap();
        assert_eq!(chain.len(), 2);
        assert!(
            chain
                .iter()
                .all(|sample| sample.units == 512 && sample.gpu > Duration::ZERO)
        );
    }
}
