use std::time::Duration;

use forja_core::{Backend, BackendError, CommandList, DType, Op, Submission};

use crate::MetalBackend;

/// Caps the temporary device allocation while keeping the probe larger than cache.
const STREAM_BYTES: u64 = 512 * 1024 * 1024;
/// Splits the probe into enough rows to expose parallel streaming throughput.
const ROWS: u32 = 256;
/// Discards cold pipeline and allocation effects before measurement.
const WARMUPS: usize = 3;
/// Bounds probe time while permitting a stable median.
const SAMPLES: usize = 9;

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
        let elements = u32::try_from(STREAM_BYTES / DType::F32.byte_size())
            .map_err(|_| BackendError::AllocationFailed)?;
        let columns = elements
            .checked_div(ROWS)
            .ok_or(BackendError::AllocationFailed)?;
        let input = self.alloc(DType::F32, &[ROWS, columns])?;
        let copied = self.alloc(DType::F32, &[ROWS, columns])?;
        let reduced = self.alloc(DType::U32, &[ROWS])?;

        let result = (|| {
            let read = measure(|| {
                let mut commands = CommandList::new();
                commands
                    .dispatch(Op::Argmax { chunk_width: 2_048 }, &[&input], &reduced)
                    .map_err(|_| BackendError::InvalidInput)?;
                gpu_time(&self.submit(commands)?)
            })?;
            let copy = measure(|| {
                let mut commands = CommandList::new();
                commands
                    .dispatch(Op::Copy, &[&input], &copied)
                    .map_err(|_| BackendError::InvalidInput)?;
                gpu_time(&self.submit(commands)?)
            })?;
            Ok(MemoryBandwidth {
                read_gigabytes_per_second: STREAM_BYTES as f64 / read.as_secs_f64() / 1e9,
                copy_gigabytes_per_second: STREAM_BYTES as f64 * 2.0 / copy.as_secs_f64() / 1e9,
            })
        })();

        self.release(&input)?;
        self.release(&copied)?;
        self.release(&reduced)?;
        result
    }
}

fn gpu_time(submission: &impl Submission) -> Result<Duration, BackendError> {
    submission.wait()?;
    submission.gpu_time().ok_or(BackendError::ExecutionFailed)
}

fn measure(
    mut sample: impl FnMut() -> Result<Duration, BackendError>,
) -> Result<Duration, BackendError> {
    for _ in 0..WARMUPS {
        sample()?;
    }
    let mut timings = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        timings.push(sample()?);
    }
    timings.sort_unstable();
    Ok(timings[SAMPLES / 2])
}

#[cfg(test)]
mod tests {
    use forja_core::Backend;

    #[test]
    fn metal_streaming_probe_reports_rates() {
        let backend = super::MetalBackend::new().unwrap();
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
