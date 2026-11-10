use std::time::Duration;

pub(super) struct Timing {
    pub decode: f64,
    pub verify: f64,
    pub plain: f64,
}

impl Timing {
    pub fn new(spans: &[u64], gpu: Duration, verify_ns: u64) -> Result<Self, String> {
        if spans.is_empty() || spans.contains(&0) {
            return Err("decode wall must be positive for every lane".to_owned());
        }
        let decode = spans.iter().copied().max().ok_or("missing decode spans")?;
        let verify = Duration::from_nanos(verify_ns);
        let plain = gpu
            .checked_sub(verify)
            .ok_or("verify GPU accounting exceeds group total")?;
        Ok(Self {
            decode: Duration::from_nanos(decode).as_secs_f64(),
            verify: verify.as_secs_f64(),
            plain: plain.as_secs_f64(),
        })
    }

    pub fn throughput(&self, output_tokens: impl Iterator<Item = u32>) -> f64 {
        output_tokens.map(f64::from).sum::<f64>() / self.decode
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_lanes_use_the_maximum_span_and_partition_gpu_work() {
        let timing = Timing::new(
            &[7_000_000_000, 3_000_000_000],
            Duration::from_secs(11),
            2_000_000_000,
        )
        .unwrap();
        assert!((timing.decode - 7.0).abs() < f64::EPSILON);
        assert!((timing.throughput([128, 33].into_iter()) - 23.0).abs() < f64::EPSILON);
        assert!((timing.verify - 2.0).abs() < f64::EPSILON);
        assert!((timing.plain - 9.0).abs() < f64::EPSILON);
        assert!(Timing::new(&[1], Duration::ZERO, 1).is_err());
        for spans in [&[][..], &[0][..], &[7, 0][..]] {
            assert!(Timing::new(spans, Duration::ZERO, 0).is_err());
        }
    }
}
