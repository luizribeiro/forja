use forja_core::{Backend, BackendDispatchData, BackendError, CommandList, DType, Op};

use crate::{CalibrationSample, MetalBackend};

const ELEMENTS: u32 = 2048 * crate::calibration::THREADS;
const GATHER_ELEMENTS: u32 = 1 << 27;
const SERIAL_ITERATIONS: u32 = 1 << 18;
const DEPENDENT_ELEMENTS: u32 = 4096;
const SERIAL_READ_ELEMENTS: u32 = 1 << 24;

#[derive(Clone, Copy)]
pub(super) enum WorkProbe {
    Scalar,
    Sfu,
    Gather,
    ScalarSerial,
    SfuSerial,
    DependentRead,
    ReadSerial,
}

impl WorkProbe {
    pub(super) const fn kernel(self) -> &'static str {
        match self {
            Self::Scalar => "calibration_scalar",
            Self::Sfu => "calibration_sfu",
            Self::Gather => "calibration_gather",
            Self::ScalarSerial => "calibration_scalar_serial",
            Self::SfuSerial => "calibration_sfu_serial",
            Self::DependentRead => "calibration_dependent_read",
            Self::ReadSerial => "calibration_read_serial",
        }
    }

    pub(super) const fn constants(self) -> [(u32, u32); 2] {
        let iterations = match self {
            Self::Gather => GATHER_ELEMENTS - 1,
            Self::ReadSerial => SERIAL_READ_ELEMENTS,
            Self::ScalarSerial | Self::SfuSerial | Self::DependentRead => SERIAL_ITERATIONS,
            _ => crate::calibration::ITERATIONS,
        };
        [(220, iterations), (221, 0)]
    }

    const fn elements(self) -> u32 {
        match self {
            Self::Gather => GATHER_ELEMENTS,
            Self::ScalarSerial | Self::SfuSerial => crate::calibration::THREADS,
            Self::DependentRead => DEPENDENT_ELEMENTS,
            Self::ReadSerial => SERIAL_READ_ELEMENTS,
            _ => ELEMENTS,
        }
    }

    fn units(self) -> Option<u64> {
        match self {
            Self::ScalarSerial => return Some(u64::from(SERIAL_ITERATIONS) * 2),
            Self::SfuSerial => return Some(u64::from(SERIAL_ITERATIONS)),
            Self::DependentRead => return Some(u64::from(SERIAL_ITERATIONS) * 4),
            Self::ReadSerial => return Some(u64::from(SERIAL_READ_ELEMENTS) * 4),
            _ => {}
        }
        u64::from(self.elements()).checked_mul(match self {
            Self::Scalar => u64::from(crate::calibration::ITERATIONS) * 2,
            Self::Sfu => u64::from(crate::calibration::ITERATIONS),
            Self::Gather => 8,
            _ => 0,
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
        for probe in [
            WorkProbe::Scalar,
            WorkProbe::Sfu,
            WorkProbe::Gather,
            WorkProbe::ScalarSerial,
            WorkProbe::SfuSerial,
            WorkProbe::DependentRead,
            WorkProbe::ReadSerial,
        ] {
            let input = self.alloc(DType::F32, &[probe.elements()])?;
            let output = match self.alloc(DType::F32, &[probe.elements()]) {
                Ok(output) => output,
                Err(error) => {
                    self.release(&input)?;
                    return Err(error);
                }
            };
            let result = (|| {
                if matches!(probe, WorkProbe::DependentRead) {
                    let cycle = (0..DEPENDENT_ELEMENTS)
                        .flat_map(|i| {
                            (i.wrapping_mul(0x9e37_79b9).wrapping_add(1) & (DEPENDENT_ELEMENTS - 1))
                                .to_le_bytes()
                        })
                        .collect::<Vec<_>>();
                    self.write(&input, &cycle)?;
                } else if !matches!(probe, WorkProbe::Gather) {
                    self.write(
                        &input,
                        &vec![
                            1.0_f32.to_le_bytes();
                            usize::try_from(probe.elements())
                                .map_err(|_| BackendError::InvalidInput)?
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
                if matches!(probe, WorkProbe::ReadSerial) {
                    let bytes = self.read(&output)?;
                    let value = bytes
                        .as_chunks::<4>()
                        .0
                        .first()
                        .copied()
                        .map(f32::from_le_bytes)
                        .ok_or(BackendError::ExecutionFailed)?;
                    if !value.is_finite() || !(1.0..1_000_000.0).contains(&value) {
                        return Err(BackendError::ExecutionFailed);
                    }
                }
                if matches!(probe, WorkProbe::DependentRead) {
                    let checksum = u32::try_from(
                        u64::from(SERIAL_ITERATIONS) * u64::from(DEPENDENT_ELEMENTS - 1) / 2,
                    )
                    .map_err(|_| BackendError::InvalidInput)?;
                    if self.read(&output)?.get(..4) != Some(checksum.to_le_bytes().as_slice()) {
                        return Err(BackendError::ExecutionFailed);
                    }
                }
                if !matches!(
                    probe,
                    WorkProbe::Gather | WorkProbe::DependentRead | WorkProbe::ReadSerial
                ) && self.read(&output)?.as_chunks::<4>().0.iter().any(|bytes| {
                    let value = f32::from_le_bytes(*bytes);
                    !value.is_finite() || !(0.5..=1.01).contains(&value)
                }) {
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
    fn dependent_reads_visit_every_address_before_repeating() {
        let mut seen = vec![false; DEPENDENT_ELEMENTS as usize];
        let mut cursor = 0_u32;
        for _ in 0..DEPENDENT_ELEMENTS {
            assert!(!seen[cursor as usize]);
            seen[cursor as usize] = true;
            cursor = cursor.wrapping_mul(0x9e37_79b9).wrapping_add(1) & (DEPENDENT_ELEMENTS - 1);
        }
        assert_eq!(cursor, 0);
        assert!(SERIAL_ITERATIONS.is_multiple_of(DEPENDENT_ELEMENTS));
    }

    #[test]
    fn metal_work_probes_execute_observable_arithmetic_and_reads() {
        let backend = MetalBackend::new().unwrap();
        let samples = backend.calibrate_work(0, 1).unwrap();
        assert_eq!(samples.len(), 7);
        for (sample, probe) in samples.iter().zip([
            WorkProbe::Scalar,
            WorkProbe::Sfu,
            WorkProbe::Gather,
            WorkProbe::ScalarSerial,
            WorkProbe::SfuSerial,
            WorkProbe::DependentRead,
            WorkProbe::ReadSerial,
        ]) {
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
