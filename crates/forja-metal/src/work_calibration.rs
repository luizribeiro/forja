use forja_core::{Backend, BackendDispatchData, BackendError, CommandList, DType, Op};

use crate::{CalibrationSample, MetalBackend};

const ELEMENTS: u32 = 2048 * crate::calibration::THREADS;
const GATHER_ELEMENTS: u32 = 1 << 27;

#[derive(Clone, Copy)]
pub(super) enum WorkProbe {
    Scalar,
    Sfu,
    Gather,
}

impl WorkProbe {
    pub(super) const fn kernel(self) -> &'static str {
        match self {
            Self::Scalar => "calibration_scalar",
            Self::Sfu => "calibration_sfu",
            Self::Gather => "calibration_gather",
        }
    }

    pub(super) const fn constants(self) -> [(u32, u32); 2] {
        let iterations = match self {
            Self::Gather => GATHER_ELEMENTS - 1,
            _ => crate::calibration::ITERATIONS,
        };
        [(220, iterations), (221, 0)]
    }

    const fn elements(self) -> u32 {
        match self {
            Self::Gather => GATHER_ELEMENTS,
            _ => ELEMENTS,
        }
    }

    fn units(self) -> Option<u64> {
        u64::from(self.elements()).checked_mul(match self {
            Self::Scalar => u64::from(crate::calibration::ITERATIONS) * 2,
            Self::Sfu => u64::from(crate::calibration::ITERATIONS),
            Self::Gather => 8,
        })
    }
}

impl MetalBackend {
    /// Measures scalar arithmetic, special functions and scattered device-memory reads.
    /// These observations calibrate work admission; they are not model benchmarks.
    ///
    /// # Errors
    /// Returns an error for invalid counts, failed allocation, compilation, execution or timing.
    pub fn calibrate_work(
        &self,
        warmups: u32,
        repetitions: u32,
    ) -> Result<Vec<CalibrationSample>, BackendError> {
        if repetitions == 0 || warmups.checked_add(repetitions).is_none() {
            return Err(BackendError::InvalidInput);
        }
        let mut samples = Vec::new();
        for probe in [WorkProbe::Scalar, WorkProbe::Sfu, WorkProbe::Gather] {
            let input = self.alloc(DType::F32, &[probe.elements()])?;
            let output = match self.alloc(DType::F32, &[probe.elements()]) {
                Ok(output) => output,
                Err(error) => {
                    self.release(&input)?;
                    return Err(error);
                }
            };
            let result = (|| {
                if !matches!(probe, WorkProbe::Gather) {
                    self.write(
                        &input,
                        &vec![
                            1.0_f32.to_le_bytes();
                            usize::try_from(ELEMENTS).map_err(|_| BackendError::InvalidInput)?
                        ]
                        .concat(),
                    )?;
                }
                let observations = self.calibration_samples(
                    probe.kernel(),
                    probe.units().ok_or(BackendError::InvalidInput)?,
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
                )?;
                if !matches!(probe, WorkProbe::Gather)
                    && self.read(&output)?.as_chunks::<4>().0.iter().any(|bytes| {
                        let value = f32::from_le_bytes(*bytes);
                        !value.is_finite() || !(0.5..=1.01).contains(&value)
                    })
                {
                    return Err(BackendError::ExecutionFailed);
                }
                Ok(observations)
            })();
            self.release(&input)?;
            self.release(&output)?;
            samples.extend(result?);
        }
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scattered_read_schedule_is_a_bijection() {
        for elements in [1_u32, 8, 32, 4096] {
            let mut visits = vec![0; elements as usize];
            for index in 0..elements {
                let address = index.wrapping_mul(0x9e37_79b9) & (elements - 1);
                visits[address as usize] += 1;
            }
            assert!(visits.iter().all(|&count| count == 1));
        }
    }

    #[test]
    fn metal_work_probes_execute_observable_arithmetic_and_reads() {
        let backend = MetalBackend::new().unwrap();
        let samples = backend.calibrate_work(0, 1).unwrap();
        assert_eq!(samples.len(), 3);
        for (sample, probe) in
            samples
                .iter()
                .zip([WorkProbe::Scalar, WorkProbe::Sfu, WorkProbe::Gather])
        {
            assert_eq!(sample.name, probe.kernel());
            assert_eq!(Some(sample.units), probe.units());
            assert!(!sample.gpu.is_zero());
        }
        assert_eq!(
            backend.calibrate_work(0, 0).unwrap_err(),
            BackendError::InvalidInput
        );
    }
}
