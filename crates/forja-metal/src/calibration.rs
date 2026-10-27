use crate::MetalBackend;
use forja_core::{Backend, BackendDispatchData, BackendError, CommandList, DType, Op, Submission};
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
    /// Whether an observation selects a configuration rather than estimating its ceiling.
    pub screening: bool,
    /// Work units executed in the timed command buffer.
    pub units: u64,
    /// Elapsed host time including submission and completion.
    pub wall: Duration,
    /// Command-buffer GPU time, without per-dispatch timestamps.
    pub gpu: Duration,
}

impl MetalBackend {
    /// Measures register MMA and threadgroup reads.
    ///
    /// Native BF16 MMA is probed; compiler refusal falls back to explicitly labelled f32 promotion.
    ///
    /// # Errors
    /// Returns an error for invalid sample counts or any allocation, execution or timing failure.
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
        let native_source = format!(
            "#define FORJA_CALIBRATION_NATIVE_BF16 1\n#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n{}",
            include_str!("calibration.metal")
        );
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        let native_bf16 = pipelines.try_calibration_shader(
            &native_source,
            Probe::NativeBf16(Chains::Four).kernel(),
            &Probe::NativeBf16(Chains::Four).constants(),
        )?;
        let f32_probe = Probe::Mma(DType::F32, Chains::Four);
        let simd_width = u32::try_from(
            pipelines
                .get(f32_probe.kernel(), &f32_probe.constants())?
                .threadExecutionWidth(),
        )
        .map_err(|_| BackendError::InvalidInput)?;
        let simdgroups = THREADS
            .checked_div(simd_width)
            .filter(|_| THREADS.is_multiple_of(simd_width))
            .ok_or(BackendError::InvalidInput)?;
        drop(pipelines);
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
                if native_bf16 {
                    Probe::NativeBf16(Chains::Four)
                } else {
                    Probe::Mma(DType::BF16, Chains::Four)
                },
                Probe::Threadgroup,
            ] {
                let units = u64::from(GROUPS)
                    * u64::from(ITERATIONS)
                    * match probe {
                        Probe::Mma(_, _) | Probe::NativeBf16(_) => {
                            u64::from(simdgroups) * 4 * 2 * 8 * 8 * 8
                        }
                        Probe::Threadgroup => u64::from(THREADS) * 16,
                    };
                let mut measured =
                    self.calibration_samples(probe.label(), units, warmups, repetitions, || {
                        let mut commands = CommandList::new();
                        commands
                            .dispatch_many_with_backend_data(
                                Op::Copy,
                                &[&input],
                                &[&output],
                                BackendDispatchData::new(probe),
                            )
                            .map_err(|_| BackendError::InvalidInput)?;
                        Ok(commands)
                    })?;
                for sample in &mut measured {
                    sample.independent_chains = probe.independent_chains();
                    sample.kernel = probe.kernel();
                    sample.arithmetic = match probe {
                        Probe::Mma(DType::BF16, _) => Some(DType::F32),
                        Probe::NativeBf16(_) => Some(DType::BF16),
                        Probe::Mma(dtype, _) => Some(dtype),
                        Probe::Threadgroup => None,
                    };
                }
                samples.extend(measured);
                let values = self.read(&output)?;
                if values.as_chunks::<4>().0.iter().copied().any(|bytes| {
                    let value = f32::from_le_bytes(bytes);
                    !value.is_finite() || value < 19.0
                }) {
                    return Err(BackendError::ExecutionFailed);
                }
            }
            Ok(samples)
        })();
        self.release(&input)?;
        self.release(&output)?;
        result
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
        assert_eq!(
            Probe::NativeBf16(Chains::Four).kernel(),
            "calibration_mma_bf16_native"
        );
        assert_ne!(
            Probe::NativeBf16(Chains::Four).kernel(),
            Probe::Mma(DType::BF16, Chains::Four).kernel()
        );
        assert_eq!(
            Probe::NativeBf16(Chains::Four).label(),
            Probe::Mma(DType::BF16, Chains::Four).label()
        );
        assert!(
            include_str!("calibration.metal")
                .contains("calibration_mma_kernel(calibration_mma_bf16_native, bfloat, bfloat)")
        );
    }

    #[test]
    fn metal_calibration_executes_observable_mma_and_threadgroup_work() {
        let backend = MetalBackend::new().unwrap();
        let samples = backend.calibrate_compute(0, 1).unwrap();
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].arithmetic, Some(DType::F32));
        assert_eq!(samples[1].arithmetic, Some(DType::F16));
        assert!(matches!(
            samples[2].arithmetic,
            Some(DType::BF16 | DType::F32)
        ));
        assert_eq!(samples[3].arithmetic, None);
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
