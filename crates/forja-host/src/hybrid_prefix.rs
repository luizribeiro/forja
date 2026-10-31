mod backing;
mod store;
pub use store::{HybridCacheUsage, HybridPrefixStore};
mod entry;
pub use entry::{HybridAttachment, HybridPrefix};

use crate::{compute, quota};

fn checked_metadata_bytes(count: u64, stride: u64, fixed: u64) -> Result<u64, compute::Error> {
    count
        .checked_mul(stride)
        .and_then(|bytes| bytes.checked_add(fixed))
        .ok_or_else(|| quota("hybrid metadata byte count overflow"))
}

/// Guest-declared snapshot ceiling reduced by an optional operator limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SnapshotAllowance {
    guest: u64,
    effective: u64,
}

impl SnapshotAllowance {
    /// Resolves a lower-only operator ceiling; zero disables snapshot retention.
    ///
    /// # Errors
    /// Refuses an operator ceiling above the guest declaration.
    pub fn new(guest: u64, operator: Option<u64>) -> Result<Self, compute::Error> {
        let effective = operator.unwrap_or(guest);
        if effective > guest {
            return Err(quota(
                "operator snapshot ceiling exceeds the guest declaration",
            ));
        }
        Ok(Self { guest, effective })
    }

    /// Effective maximum bytes retained for cached immutable state and its metadata.
    #[must_use]
    pub const fn bytes(self) -> u64 {
        self.effective
    }

    /// Changes the operator ceiling while retaining the original guest declaration.
    ///
    /// # Errors
    /// Refuses increases above the guest declaration.
    pub fn with_operator(self, bytes: u64) -> Result<Self, compute::Error> {
        Self::new(self.guest, Some(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn metadata_extents_check_both_multiplication_and_addition() {
        assert_eq!(checked_metadata_bytes(7, 64, 256).unwrap(), 704);
        assert_eq!(checked_metadata_bytes(0, u64::MAX, 33).unwrap(), 33);
        assert_eq!(checked_metadata_bytes(1, u64::MAX, 0).unwrap(), u64::MAX);
        assert!(checked_metadata_bytes(u64::MAX, 2, 0).is_err());
        assert!(checked_metadata_bytes(1, u64::MAX, 1).is_err());
    }
    #[test]
    fn operator_can_only_lower_the_declared_snapshot_allowance() {
        let allowance = SnapshotAllowance::new(4097, None).unwrap();
        assert_eq!(allowance.bytes(), 4097);
        assert_eq!(allowance.with_operator(33).unwrap().bytes(), 33);
        assert_eq!(allowance.with_operator(0).unwrap().bytes(), 0);
        assert!(allowance.with_operator(4098).is_err());
        assert!(SnapshotAllowance::new(0, Some(1)).is_err());
        assert_eq!(allowance.bytes(), 4097);
    }
}
