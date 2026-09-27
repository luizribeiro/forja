use std::{error::Error, time::Duration};

use forja_host::Limits;

pub(crate) fn argmax(values: &[f32]) -> Result<u32, Box<dyn Error>> {
    let index = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or("cannot take argmax of empty logits")?;
    Ok(u32::try_from(index)?)
}

pub(crate) const fn limits() -> Limits {
    Limits::new(
        8 * 1024 * 1024 * 1024,
        4,
        1_000_000_000,
        20_000,
        1024 * 1024 * 1024,
    )
    .with_command_limits(4_096, u64::MAX)
    .with_guest_call_timeout(Duration::from_secs(300))
    .with_gpu_limits(Duration::from_secs(60), Duration::from_secs(3_600))
}
