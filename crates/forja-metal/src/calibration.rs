use crate::MetalBackend;
use forja_core::{Backend, BackendDispatchData, BackendError, CommandList, DType, Op, Submission};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub(crate) enum Probe {
    Mma(DType),
    Threadgroup,
}

impl Probe {
    pub(crate) const fn kernel(self) -> &'static str {
        match self {
            Self::Mma(DType::F16) => "calibration_mma_f16",
            Self::Mma(DType::BF16) => "calibration_mma_bf16",
            Self::Mma(_) => "calibration_mma_f32",
            Self::Threadgroup => "calibration_threadgroup",
        }
    }
}

const GROUPS: u32 = 2048;
const THREADS: u32 = 128;
pub(crate) const ITERATIONS: u32 = 4096;

/// One unprofiled device calibration observation, never a model benchmark.
#[derive(Clone, Debug)]
pub struct CalibrationSample {
    /// Probe identity and units (bytes, FLOPs, or dispatches).
    pub name: &'static str,
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
    /// BF16 operands are promoted to f32 MMA, matching the supported shader path.
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
        let mut samples = Vec::new();
        let result = (|| {
            for probe in [
                Probe::Mma(DType::F32),
                Probe::Mma(DType::F16),
                Probe::Mma(DType::BF16),
                Probe::Threadgroup,
            ] {
                let units = u64::from(GROUPS)
                    * u64::from(ITERATIONS)
                    * match probe {
                        Probe::Mma(_) => 4 * 4 * 2 * 8 * 8 * 8,
                        Probe::Threadgroup => u64::from(THREADS) * 16,
                    };
                samples.extend(self.calibration_samples(
                    probe.kernel(),
                    units,
                    warmups,
                    repetitions,
                    || {
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
                    },
                )?);
                let values = self.read(&output)?;
                if values.as_chunks::<4>().0.iter().copied().any(|bytes| {
                    let value = f32::from_le_bytes(bytes);
                    !value.is_finite() || value < 19.0
                }) {
                    return Err(BackendError::ExecutionFailed);
                }
            }
            Ok(())
        })();
        self.release(&input)?;
        self.release(&output)?;
        result?;
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
                    name,
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
    fn metal_calibration_executes_observable_mma_and_threadgroup_work() {
        let backend = MetalBackend::new().unwrap();
        let samples = backend.calibrate_compute(0, 1).unwrap();
        assert_eq!(samples.len(), 4);
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
