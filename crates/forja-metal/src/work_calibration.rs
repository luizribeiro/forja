use forja_core::{Backend, BackendDispatchData, BackendError, CommandList, DType, Op};

use crate::{CalibrationSample, MetalBackend};

const ELEMENTS: u32 = 2048 * crate::calibration::THREADS;
const GATHER_ELEMENTS: u32 = 1 << 27;
const SERIAL_ITERATIONS: u32 = 1 << 18;
const DEPENDENT_ELEMENTS: u32 = 4096;
const SERIAL_READ_ELEMENTS: u32 = 1 << 24;
const DISPATCH_GROUPS: u32 = 1 << 18;

#[derive(Clone, Copy)]
pub(super) enum WorkProbe {
    Scalar,
    Sfu,
    Gather,
    ScalarSerial,
    SfuSerial,
    DependentRead,
    ReadSerial,
    ScalarGroup,
    SfuGroup,
    ReadGroup,
    GatherGroup,
    GroupDispatch,
}

impl WorkProbe {
    const ALL: [Self; 12] = [
        Self::Scalar,
        Self::Sfu,
        Self::Gather,
        Self::ScalarSerial,
        Self::SfuSerial,
        Self::DependentRead,
        Self::ReadSerial,
        Self::ScalarGroup,
        Self::SfuGroup,
        Self::ReadGroup,
        Self::GatherGroup,
        Self::GroupDispatch,
    ];

    pub(super) const fn kernel(self) -> &'static str {
        match self {
            Self::Scalar => "calibration_scalar",
            Self::Sfu => "calibration_sfu",
            Self::Gather => "calibration_gather",
            Self::ScalarSerial => "calibration_scalar_serial",
            Self::SfuSerial => "calibration_sfu_serial",
            Self::DependentRead => "calibration_dependent_read",
            Self::ReadSerial => "calibration_read_serial",
            Self::ScalarGroup => "calibration_scalar_group",
            Self::SfuGroup => "calibration_sfu_group",
            Self::ReadGroup => "calibration_read_group",
            Self::GatherGroup => "calibration_gather_group",
            Self::GroupDispatch => "calibration_group_dispatch",
        }
    }

    pub(super) const fn constants(self) -> [(u32, u32); 2] {
        let iterations = match self {
            Self::Gather => GATHER_ELEMENTS - 1,
            Self::GroupDispatch => 16,
            Self::ReadSerial | Self::ReadGroup => SERIAL_READ_ELEMENTS,
            Self::ScalarSerial
            | Self::SfuSerial
            | Self::DependentRead
            | Self::ScalarGroup
            | Self::SfuGroup
            | Self::GatherGroup => SERIAL_ITERATIONS,
            _ => crate::calibration::ITERATIONS,
        };
        [(220, iterations), (221, 0)]
    }

    const fn elements(self) -> u32 {
        match self {
            Self::Gather => GATHER_ELEMENTS,
            Self::GroupDispatch => DISPATCH_GROUPS * crate::calibration::THREADS,
            Self::ScalarSerial | Self::SfuSerial | Self::ScalarGroup | Self::SfuGroup => {
                crate::calibration::THREADS
            }
            Self::DependentRead => DEPENDENT_ELEMENTS,
            Self::ReadSerial | Self::ReadGroup | Self::GatherGroup => SERIAL_READ_ELEMENTS,
            _ => ELEMENTS,
        }
    }

    fn dependent_lanes(self) -> Option<u32> {
        match self {
            Self::DependentRead => Some(1),
            Self::GatherGroup => Some(32),
            _ => None,
        }
    }

    fn units(self) -> Option<u64> {
        match self {
            Self::GroupDispatch => return Some(u64::from(DISPATCH_GROUPS)),
            Self::ScalarSerial => return Some(u64::from(SERIAL_ITERATIONS) * 2),
            Self::SfuSerial => return Some(u64::from(SERIAL_ITERATIONS)),
            Self::DependentRead => return Some(u64::from(SERIAL_ITERATIONS) * 4),
            Self::ReadSerial | Self::ReadGroup => return Some(u64::from(SERIAL_READ_ELEMENTS) * 4),
            Self::ScalarGroup => return Some(u64::from(SERIAL_ITERATIONS) * 32 * 2),
            Self::SfuGroup => return Some(u64::from(SERIAL_ITERATIONS) * 32),
            Self::GatherGroup => return Some(u64::from(SERIAL_ITERATIONS) * 32 * 4),
            _ => {}
        }
        u64::from(self.elements()).checked_mul(match self {
            Self::Scalar => u64::from(crate::calibration::ITERATIONS) * 2,
            Self::Sfu => u64::from(crate::calibration::ITERATIONS),
            Self::Gather => 8,
            _ => 0,
        })
    }
    fn verify(
        self,
        backend: &MetalBackend,
        output: &forja_core::Tensor,
    ) -> Result<(), BackendError> {
        if matches!(self, Self::Gather) {
            return Ok(());
        }
        let probe = self;
        let bytes = backend.read(output)?;
        if matches!(probe, WorkProbe::ReadSerial | WorkProbe::ReadGroup) {
            let lanes = if matches!(probe, WorkProbe::ReadGroup) {
                32
            } else {
                1
            };
            let values = bytes
                .as_chunks::<4>()
                .0
                .get(..lanes)
                .ok_or(BackendError::ExecutionFailed)?;
            if values
                .iter()
                .map(|v| f32::from_le_bytes(*v))
                .any(|v| !v.is_finite() || !(1_000.0..1_000_000.0).contains(&v))
            {
                return Err(BackendError::ExecutionFailed);
            }
        }
        if let Some(lanes) = probe.dependent_lanes() {
            let expected = (0..lanes)
                .flat_map(|start| {
                    let mut cursor = start;
                    let mut sum = 0_u32;
                    for _ in 0..SERIAL_ITERATIONS {
                        cursor = cursor.wrapping_mul(0x9e37_79b9).wrapping_add(1)
                            & (probe.elements() - 1);
                        sum = sum.wrapping_add(cursor);
                    }
                    sum.to_le_bytes()
                })
                .collect::<Vec<_>>();
            if bytes.get(..expected.len()) != Some(expected.as_slice()) {
                return Err(BackendError::ExecutionFailed);
            }
        }
        if !matches!(
            probe,
            WorkProbe::Gather
                | WorkProbe::DependentRead
                | WorkProbe::ReadSerial
                | WorkProbe::ReadGroup
                | WorkProbe::GatherGroup
        ) && bytes.as_chunks::<4>().0.iter().any(|bytes| {
            let value = f32::from_le_bytes(*bytes);
            !value.is_finite() || !(0.5..=1.01).contains(&value)
        }) {
            return Err(BackendError::ExecutionFailed);
        }
        Ok(())
    }
}

impl MetalBackend {
    /// Measures scalar arithmetic, special functions, memory and threadgroup overhead.
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
        for probe in WorkProbe::ALL {
            let input = self.alloc(DType::F32, &[probe.elements()])?;
            let output = match self.alloc(DType::F32, &[probe.elements()]) {
                Ok(output) => output,
                Err(error) => {
                    self.release(&input)?;
                    return Err(error);
                }
            };
            let result: Result<_, BackendError> = (|| {
                if probe.dependent_lanes().is_some() {
                    let cycle = (0..probe.elements())
                        .flat_map(|i| {
                            (i.wrapping_mul(0x9e37_79b9).wrapping_add(1) & (probe.elements() - 1))
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
                probe.verify(self, &output)?;
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
        assert_eq!(samples.len(), WorkProbe::ALL.len());
        for (sample, probe) in samples.iter().zip(WorkProbe::ALL) {
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
