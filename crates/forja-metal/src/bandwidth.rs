use forja_core::{Backend, BackendError, CommandList, DType, Op};

use crate::{CalibrationSample, MetalBackend};

/// Caps the temporary device allocation while keeping the probe larger than cache.
const STREAM_BYTES: u64 = 512 * 1024 * 1024;
/// Splits the probe into enough rows to expose parallel streaming throughput.
const ROWS: u32 = 256;
/// Discards cold pipeline and allocation effects before measurement.
const WARMUPS: u32 = 3;
/// Bounds probe time while permitting a stable median.
const SAMPLES: u32 = 9;

/// Sustained device-memory rates from large streaming kernels.
#[derive(Clone, Copy, Debug)]
pub struct MemoryBandwidth {
    /// Read-only reduction throughput in decimal gigabytes per second.
    pub read_gigabytes_per_second: f64,
    /// Combined read and write traffic from a copy in decimal gigabytes per second.
    pub copy_gigabytes_per_second: f64,
}

impl MetalBackend {
    /// Measures sustained device-memory bandwidth with a read-only reduction and a copy.
    ///
    /// # Errors
    ///
    /// Returns a backend error when allocation, submission, timing, or release fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn measure_memory_bandwidth(&self) -> Result<MemoryBandwidth, BackendError> {
        let mut cached = self
            .bandwidth
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        if let Some(bandwidth) = *cached {
            return Ok(bandwidth);
        }
        let bandwidth = self.measure_memory_bandwidth_uncached()?;
        *cached = Some(bandwidth);
        Ok(bandwidth)
    }

    #[allow(clippy::cast_precision_loss)]
    fn measure_memory_bandwidth_uncached(&self) -> Result<MemoryBandwidth, BackendError> {
        let samples = self.calibrate_memory(WARMUPS, SAMPLES)?;
        let median = |name| {
            let mut times = samples
                .iter()
                .filter(|sample| sample.name == name)
                .map(|sample| sample.gpu)
                .collect::<Vec<_>>();
            times.sort_unstable();
            times[times.len() / 2].as_secs_f64()
        };
        Ok(MemoryBandwidth {
            read_gigabytes_per_second: STREAM_BYTES as f64 / median("streaming_read") / 1e9,
            copy_gigabytes_per_second: STREAM_BYTES as f64 * 2.0 / median("streaming_copy") / 1e9,
        })
    }

    /// Records raw streaming read/copy timings for device calibration.
    ///
    /// # Errors
    /// Returns an error on invalid counts or allocation, execution, timing or release failure.
    pub fn calibrate_memory(
        &self,
        warmups: u32,
        repetitions: u32,
    ) -> Result<Vec<CalibrationSample>, BackendError> {
        let columns = u32::try_from(STREAM_BYTES / 4 / u64::from(ROWS))
            .map_err(|_| BackendError::AllocationFailed)?;
        let input = self.alloc(DType::F32, &[ROWS, columns])?;
        let copied = self.alloc(DType::F32, &[ROWS, columns])?;
        let reduced = self.alloc(DType::U32, &[ROWS])?;
        let result = (|| {
            let mut samples = self.calibration_samples(
                "streaming_read",
                STREAM_BYTES,
                warmups,
                repetitions,
                || {
                    let mut commands = CommandList::new();
                    commands
                        .dispatch(Op::Argmax { chunk_width: 2048 }, &[&input], &reduced)
                        .map_err(|_| BackendError::InvalidInput)?;
                    Ok(commands)
                },
            )?;
            samples.extend(self.calibration_samples(
                "streaming_copy",
                STREAM_BYTES * 2,
                warmups,
                repetitions,
                || {
                    let mut commands = CommandList::new();
                    commands
                        .dispatch(Op::Copy, &[&input], &copied)
                        .map_err(|_| BackendError::InvalidInput)?;
                    Ok(commands)
                },
            )?);
            Ok(samples)
        })();
        self.release(&input)?;
        self.release(&copied)?;
        self.release(&reduced)?;
        result
    }
}

#[cfg(test)]
mod tests {
    use forja_core::Backend;

    #[test]
    fn metal_streaming_probe_reports_rates() {
        let backend = super::MetalBackend::new().unwrap();
        let samples = backend.calibrate_memory(0, 1).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].units, super::STREAM_BYTES);
        assert_eq!(samples[1].units, super::STREAM_BYTES * 2);
        let bandwidth = backend.measure_memory_bandwidth().unwrap();
        assert!((0.0..10_000.0).contains(&bandwidth.read_gigabytes_per_second));
        assert!((0.0..10_000.0).contains(&bandwidth.copy_gigabytes_per_second));
    }

    #[test]
    fn metal_device_info_reports_capabilities_and_reuses_bandwidth() {
        let backend = super::MetalBackend::new().unwrap();
        let first = backend.device_info().unwrap();
        let second = backend.device_info().unwrap();

        assert!(first.family.starts_with("apple"));
        assert!(first.recommended_max_working_set_size > 0);
        assert!(first.max_threadgroup_memory_length > 0);
        assert!(first.simd_width > 0);
        assert_eq!(first, second);
    }
}
