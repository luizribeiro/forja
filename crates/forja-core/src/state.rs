//! Validated geometry for opaque per-sequence state.

mod block;
pub use block::DeltaBlockScratch;
mod access;
pub use access::StateAccess;
mod ops;
pub use ops::StateOp;
mod pool;
pub use pool::{StatePool, StateSequence, StateSlot, StateTransaction};

use crate::{BackendError, DType, MAX_RANK, PagedLayout, PagedPlane};

/// One independently typed, fixed-size state plane.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StatePlane {
    dtype: DType,
    shape: Vec<u32>,
    elements: u64,
}

impl StatePlane {
    /// Validates a nonempty floating-point plane and its checked byte size.
    ///
    /// # Errors
    /// Returns invalid input for bad geometry, dtype or overflow.
    pub fn new(dtype: DType, shape: Vec<u32>) -> Result<Self, BackendError> {
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
            || shape.is_empty()
            || shape.len() > MAX_RANK
            || shape.contains(&0)
        {
            return Err(BackendError::InvalidInput);
        }
        let elements = shape.iter().try_fold(1_u64, |size, &dimension| {
            size.checked_mul(u64::from(dimension))
                .ok_or(BackendError::InvalidInput)
        })?;
        elements
            .checked_mul(dtype.byte_size())
            .ok_or(BackendError::InvalidInput)?;
        Ok(Self {
            dtype,
            shape,
            elements,
        })
    }

    /// Stored scalar type.
    #[must_use]
    pub const fn dtype(&self) -> DType {
        self.dtype
    }

    /// Logical shape of one sequence version.
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// Checked number of scalars per version.
    #[must_use]
    pub const fn elements(&self) -> u64 {
        self.elements
    }
}

/// Immutable geometry shared by every version in one state pool.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateLayout {
    planes: Vec<StatePlane>,
    backing: PagedLayout,
    bytes: u64,
}

impl StateLayout {
    /// Validates the planes and their aggregate byte size.
    ///
    /// # Errors
    /// Returns invalid input for too many planes or overflowing allocation sizes.
    pub fn new(planes: Vec<StatePlane>) -> Result<Self, BackendError> {
        let bytes = planes.iter().try_fold(0_u64, |size, plane| {
            size.checked_add(plane.elements * plane.dtype.byte_size())
                .ok_or(BackendError::InvalidInput)
        })?;
        let backing = PagedLayout::new(
            1,
            planes
                .iter()
                .enumerate()
                .map(|(index, plane)| PagedPlane {
                    dtype: plane.dtype,
                    elements_per_token: plane.elements,
                    alignment: plane.dtype.byte_size(),
                    label: format!("state-{index}"),
                })
                .collect(),
        )
        .map_err(|_| BackendError::InvalidInput)?;
        Ok(Self {
            planes,
            backing,
            bytes,
        })
    }

    /// Validated plane descriptors.
    #[must_use]
    pub fn planes(&self) -> &[StatePlane] {
        &self.planes
    }

    /// Bytes occupied by one complete sequence version.
    #[must_use]
    pub const fn version_bytes(&self) -> u64 {
        self.bytes
    }

    /// Equivalent fixed-record storage geometry for the trusted allocator.
    #[must_use]
    pub fn storage_layout(&self) -> &PagedLayout {
        &self.backing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_geometry_is_checked_once() {
        for shape in [vec![], vec![0], vec![u32::MAX; 3]] {
            assert!(StatePlane::new(DType::F32, shape).is_err());
        }
        assert!(StatePlane::new(DType::U32, vec![7]).is_err());
        assert!(StateLayout::new(vec![]).is_err());
        let layout = StateLayout::new(vec![
            StatePlane::new(DType::F32, vec![7, 33]).unwrap(),
            StatePlane::new(DType::BF16, vec![3, 4097]).unwrap(),
        ])
        .unwrap();
        assert_eq!(layout.version_bytes(), 7 * 33 * 4 + 3 * 4097 * 2);
        assert_eq!(layout.storage_layout().planes().len(), 2);
    }
}
