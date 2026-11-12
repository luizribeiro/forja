use forja_core::{Backend, BackendDispatchData, BackendError, CommandList, DType, Op};

use crate::{CalibrationSample, MetalBackend};

const ELEMENTS: u32 = 2048 * crate::calibration::THREADS;
const GATHER_ELEMENTS: u32 = 1 << 27;
const SERIAL_ITERATIONS: u32 = 1 << 18;
const DEPENDENT_ELEMENTS: u32 = 4096;
const SERIAL_READ_ELEMENTS: u32 = 1 << 24;
const DISPATCH_GROUPS: u32 = 1 << 18;
const REUSED_ELEMENTS: u32 = 1 << 25;
pub(super) const REUSED_READ_BYTES: u64 = (REUSED_ELEMENTS as u64) * 4;

#[derive(Clone, Copy)]
pub(super) enum WorkProbe {
    Scalar,
    ScalarParallel,
    ScalarWideGroup,
    SfuWideGroup,
    Sfu,
    Gather,
    ReusedRead,
    SingleDispatch,
    ScalarSerial,
    SfuSerial,
    DependentRead,
    ReadSerial,
    ScalarGroup,
    SfuGroup,
    ReadGroup,
    ReadWideGroup,
    GatherGroup,
    GroupDispatch,
}

impl WorkProbe {
    const ALL: [Self; 14] = [
        Self::Scalar,
        Self::Sfu,
        Self::Gather,
        Self::ReusedRead,
        Self::SingleDispatch,
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
            Self::ScalarParallel => "calibration_scalar_parallel",
            Self::ScalarWideGroup => "calibration_scalar_wide_group",
            Self::SfuWideGroup => "calibration_sfu_wide_group",
            Self::Sfu => "calibration_sfu",
            Self::Gather => "calibration_gather",
            Self::ReusedRead => "calibration_reused_read",
            Self::SingleDispatch => "calibration_single_dispatch",
            Self::ScalarSerial => "calibration_scalar_serial",
            Self::SfuSerial => "calibration_sfu_serial",
            Self::DependentRead => "calibration_dependent_read",
            Self::ReadSerial => "calibration_read_serial",
            Self::ScalarGroup => "calibration_scalar_group",
            Self::SfuGroup => "calibration_sfu_group",
            Self::ReadGroup => "calibration_read_group",
            Self::ReadWideGroup => "calibration_read_wide_group",
            Self::GatherGroup => "calibration_gather_group",
            Self::GroupDispatch => "calibration_group_dispatch",
        }
    }

    pub(super) const fn constants(self) -> [(u32, u32); 2] {
        let iterations = match self {
            Self::Gather => GATHER_ELEMENTS - 1,
            Self::ReusedRead => REUSED_ELEMENTS - 1,
            Self::SingleDispatch => 1,
            Self::GroupDispatch => 16,
            Self::ReadSerial | Self::ReadGroup | Self::ReadWideGroup => SERIAL_READ_ELEMENTS,
            Self::ScalarWideGroup
            | Self::SfuWideGroup
            | Self::ScalarSerial
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
            Self::ReusedRead => REUSED_ELEMENTS,
            Self::GroupDispatch => DISPATCH_GROUPS * crate::calibration::THREADS,
            Self::SingleDispatch
            | Self::ScalarWideGroup
            | Self::SfuWideGroup
            | Self::ScalarSerial
            | Self::SfuSerial
            | Self::ScalarGroup
            | Self::SfuGroup => crate::calibration::THREADS,
            Self::DependentRead => DEPENDENT_ELEMENTS,
            Self::ReadSerial | Self::ReadGroup | Self::ReadWideGroup | Self::GatherGroup => {
                SERIAL_READ_ELEMENTS
            }
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
            Self::ScalarParallel => {
                return Some(
                    u64::from(ELEMENTS) * u64::from(crate::calibration::ITERATIONS) * 8 * 2,
                );
            }
            Self::ScalarWideGroup => {
                return Some(
                    u64::from(SERIAL_ITERATIONS) * u64::from(crate::calibration::THREADS) * 2,
                );
            }
            Self::SfuWideGroup => {
                return Some(u64::from(SERIAL_ITERATIONS) * u64::from(crate::calibration::THREADS));
            }
            Self::GroupDispatch => return Some(u64::from(DISPATCH_GROUPS)),
            Self::ReusedRead => return Some(REUSED_READ_BYTES * 32),
            Self::SingleDispatch => return Some(1),
            Self::ScalarSerial => return Some(u64::from(SERIAL_ITERATIONS) * 2),
            Self::SfuSerial => return Some(u64::from(SERIAL_ITERATIONS)),
            Self::DependentRead => return Some(u64::from(SERIAL_ITERATIONS) * 4),
            Self::ReadSerial | Self::ReadGroup | Self::ReadWideGroup => {
                return Some(u64::from(SERIAL_READ_ELEMENTS) * 4);
            }
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
        if matches!(
            probe,
            WorkProbe::ReadSerial | WorkProbe::ReadGroup | WorkProbe::ReadWideGroup
        ) {
            let lanes = if matches!(probe, WorkProbe::ReadGroup | WorkProbe::ReadWideGroup) {
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
                .any(|v| !v.is_finite() || !(1_000.0..20_000_000.0).contains(&v))
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
                | WorkProbe::ReadWideGroup
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
        self.calibrate_work_probes(&WorkProbe::ALL, warmups, repetitions)
    }

    pub(super) fn calibrate_work_group(
        &self,
        repetitions: u32,
    ) -> Result<Vec<CalibrationSample>, BackendError> {
        self.calibrate_work_probes(
            &[
                WorkProbe::ReadWideGroup,
                WorkProbe::ScalarParallel,
                WorkProbe::ScalarWideGroup,
                WorkProbe::SfuWideGroup,
            ],
            0,
            repetitions,
        )
    }

    fn calibrate_work_probes(
        &self,
        probes: &[WorkProbe],
        warmups: u32,
        repetitions: u32,
    ) -> Result<Vec<CalibrationSample>, BackendError> {
        if repetitions == 0 || warmups.checked_add(repetitions).is_none() {
            return Err(BackendError::InvalidInput);
        }
        let mut samples = Vec::new();
        for &probe in probes {
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
    fn metal_private_admission_probes_verify_arithmetic_and_traffic() {
        let backend = MetalBackend::new().unwrap();
        let samples = backend.calibrate_compute_group(1).unwrap();
        assert_eq!(samples.len(), 3);
        for sample in samples {
            assert_eq!(sample.name, "calibration_mma_group");
            assert_eq!(
                sample.units,
                4 * 4 * 2 * 8 * 8 * 8 * u64::from(crate::calibration::ITERATIONS)
            );
            assert!(!sample.gpu.is_zero());
        }
        let samples = backend.calibrate_work_group(1).unwrap();
        assert_eq!(samples.len(), 4);
        assert_eq!(samples[0].name, "calibration_read_wide_group");
        assert_eq!(samples[0].units, u64::from(SERIAL_READ_ELEMENTS) * 4);
        for (sample, probe) in samples.iter().zip([
            WorkProbe::ReadWideGroup,
            WorkProbe::ScalarParallel,
            WorkProbe::ScalarWideGroup,
            WorkProbe::SfuWideGroup,
        ]) {
            assert_eq!(sample.name, probe.kernel());
            assert_eq!(Some(sample.units), probe.units());
            assert!(!sample.gpu.is_zero());
        }
    }

    #[test]
    fn repeated_read_passes_visit_the_complete_declared_footprint() {
        assert_eq!(WorkProbe::ReusedRead.units(), Some(REUSED_READ_BYTES * 32));
        for pass in 0..32_u64 {
            let offset = pass * 1_048_576;
            let addresses = (0..u64::from(REUSED_ELEMENTS))
                .step_by(4096)
                .map(|index| (index + offset) & u64::from(REUSED_ELEMENTS - 1))
                .collect::<std::collections::BTreeSet<_>>();
            assert_eq!(
                addresses.len(),
                usize::try_from(REUSED_ELEMENTS / 4096).unwrap()
            );
        }
    }

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
