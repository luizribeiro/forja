use std::{error::Error, time::Duration};

use forja_host::Limits;

/// Selects the last greatest value under the IEEE total order.
///
/// Equal values select the later index, and positive NaNs sort above finite values rather than
/// being ignored.
pub(crate) fn argmax(values: &[f32]) -> Result<u32, Box<dyn Error>> {
    let index = values
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or("cannot take argmax of empty logits")?;
    Ok(u32::try_from(index)?)
}

pub(crate) fn read_token(bytes: &[u8]) -> Result<u32, Box<dyn Error>> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| "selected token must contain exactly four bytes")?;
    Ok(u32::from_le_bytes(bytes))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_rejects_empty_values() {
        assert_eq!(
            argmax(&[]).unwrap_err().to_string(),
            "cannot take argmax of empty logits"
        );
    }

    #[test]
    fn argmax_selects_the_last_tied_value() -> Result<(), Box<dyn Error>> {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0])?, 2);
        Ok(())
    }

    #[test]
    fn argmax_uses_total_order_for_nan() -> Result<(), Box<dyn Error>> {
        assert_eq!(argmax(&[1.0, f32::NAN, 2.0])?, 1);
        Ok(())
    }

    #[test]
    fn reads_one_selected_token() -> Result<(), Box<dyn Error>> {
        assert_eq!(read_token(&7_u32.to_le_bytes())?, 7);
        assert!(read_token(&[0; 8]).is_err());
        Ok(())
    }
}
