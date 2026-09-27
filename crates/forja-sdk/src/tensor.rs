use std::marker::PhantomData;

use crate::{Element, Error, Result, sys};

/// A strided selection along one tensor axis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Slice {
    start: u32,
    len: u32,
    step: u32,
}

impl Slice {
    /// Creates a slice selecting `len` elements from `start` at `step` intervals.
    ///
    /// # Errors
    ///
    /// Returns an error when `step` is zero.
    pub fn new(start: u32, len: u32, step: u32) -> Result<Self> {
        if step == 0 {
            return Err(Error::new("slice step must be at least one"));
        }
        Ok(Self { start, len, step })
    }
}

/// A typed tensor owned by the Forja host.
pub struct Tensor<T: Element> {
    handle: sys::Handle,
    shape: Vec<u32>,
    element: PhantomData<T>,
}

impl<T: Element> Tensor<T> {
    /// Allocates a contiguous tensor and initializes it from a slice.
    ///
    /// # Errors
    ///
    /// Returns an error when the shape size overflows, the value count differs,
    /// or the host refuses the allocation or write.
    pub fn from_slice(values: &[T], shape: &[u32]) -> Result<Self> {
        if element_count(shape)? != u64::try_from(values.len()).map_err(|_| size_error())? {
            return Err(Error::new("data length does not match tensor shape"));
        }
        let handle = sys::alloc(T::DTYPE, shape)?;
        sys::write(&handle, &T::encode(values))?;
        Ok(Self::from_handle(handle, shape.to_vec()))
    }

    /// Returns the extent of each tensor axis.
    #[must_use]
    pub fn shape(&self) -> &[u32] {
        &self.shape
    }

    /// Selects a strided range independently on every axis.
    ///
    /// # Errors
    ///
    /// Returns an error for a rank mismatch or a host-refused view.
    pub fn slice(&self, slices: &[Slice]) -> Result<Self> {
        if slices.len() != self.shape.len() {
            return Err(Error::new("slice rank does not match tensor rank"));
        }
        let operation = sys::View::Slice(
            slices
                .iter()
                .map(|slice| sys::Slice {
                    start: slice.start,
                    len: slice.len,
                    step: slice.step,
                })
                .collect(),
        );
        self.view(operation, slices.iter().map(|slice| slice.len).collect())
    }

    /// Selects a contiguous range on one axis.
    ///
    /// # Errors
    ///
    /// Returns an error when the axis or range is invalid.
    pub fn narrow(&self, axis: usize, start: u32, len: u32) -> Result<Self> {
        let extent = self
            .shape
            .get(axis)
            .copied()
            .ok_or_else(|| Error::new("narrow axis is out of range"))?;
        if start.checked_add(len).is_none_or(|end| end > extent) {
            return Err(Error::new("narrow range exceeds its axis"));
        }
        let mut slices = self
            .shape
            .iter()
            .map(|&extent| Slice {
                start: 0,
                len: extent,
                step: 1,
            })
            .collect::<Vec<_>>();
        slices[axis] = Slice {
            start,
            len,
            step: 1,
        };
        self.slice(&slices)
    }

    /// Reorders the tensor axes.
    ///
    /// # Errors
    ///
    /// Returns an error when the axes are not a valid permutation.
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        if axes.len() != self.shape.len() {
            return Err(Error::new("permutation rank does not match tensor rank"));
        }
        let mut seen = vec![false; axes.len()];
        let axes = axes
            .iter()
            .map(|&axis| {
                let slot = seen
                    .get_mut(axis)
                    .ok_or_else(|| Error::new("axis is out of range"))?;
                if *slot {
                    return Err(Error::new("axis appears more than once"));
                }
                *slot = true;
                u8::try_from(axis).map_err(|_| Error::new("axis is out of range"))
            })
            .collect::<Result<Vec<_>>>()?;
        let shape = axes
            .iter()
            .map(|&axis| self.shape[usize::from(axis)])
            .collect();
        self.view(sys::View::Permute(axes), shape)
    }

    /// Swaps the last two axes.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor has fewer than two axes.
    pub fn t(&self) -> Result<Self> {
        if self.shape.len() < 2 {
            return Err(Error::new("transpose requires at least two axes"));
        }
        let mut axes = (0..self.shape.len()).collect::<Vec<_>>();
        let rank = axes.len();
        axes.swap(rank - 2, rank - 1);
        self.permute(&axes)
    }

    /// Changes the shape without copying contiguous storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the view is non-contiguous or sizes differ.
    pub fn reshape(&self, shape: &[u32]) -> Result<Self> {
        self.view(sys::View::Reshape(shape.to_vec()), shape.to_vec())
    }

    /// Broadcasts size-one axes to a target shape.
    ///
    /// # Errors
    ///
    /// Returns an error when the source cannot broadcast to the target.
    pub fn broadcast_as(&self, shape: &[u32]) -> Result<Self> {
        self.view(sys::View::Broadcast(shape.to_vec()), shape.to_vec())
    }

    /// Gathers the logical tensor values.
    ///
    /// # Errors
    ///
    /// Returns an error when reading fails.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        T::decode(&sys::read(&self.handle)?)
    }

    fn view(&self, operation: sys::View, shape: Vec<u32>) -> Result<Self> {
        let handle = sys::view(&self.handle, operation)?;
        Ok(Self::from_handle(handle, shape))
    }

    pub(crate) fn from_handle(handle: sys::Handle, shape: Vec<u32>) -> Self {
        Self {
            handle,
            shape,
            element: PhantomData,
        }
    }
}

fn element_count(shape: &[u32]) -> Result<u64> {
    shape.iter().try_fold(1_u64, |count, &extent| {
        count.checked_mul(u64::from(extent)).ok_or_else(size_error)
    })
}

fn size_error() -> Error {
    Error::new("tensor element count overflowed")
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    #[test]
    fn native_views_round_trip_non_contiguous_values() {
        let tensor = Tensor::from_slice(&(0_u32..21).collect::<Vec<_>>(), &[3, 7]).unwrap();
        let view = tensor
            .slice(&[Slice::new(0, 3, 1).unwrap(), Slice::new(1, 3, 2).unwrap()])
            .unwrap()
            .t()
            .unwrap();

        assert_eq!(view.shape(), [3, 3]);
        assert_eq!(view.to_vec().unwrap(), [1, 8, 15, 3, 10, 17, 5, 12, 19]);
    }

    #[test]
    fn rejects_invalid_slice_preconditions() {
        let tensor = Tensor::from_slice(&(0_u32..6).collect::<Vec<_>>(), &[2, 3]).unwrap();

        assert!(Slice::new(0, 1, 0).is_err());
        assert!(tensor.slice(&[Slice::new(0, 1, 1).unwrap()]).is_err());
    }

    #[test]
    fn rejects_invalid_narrow_preconditions() {
        let tensor = Tensor::from_slice(&(0_u32..6).collect::<Vec<_>>(), &[2, 3]).unwrap();

        assert!(tensor.narrow(2, 0, 1).is_err());
        assert!(tensor.narrow(1, 2, 2).is_err());
        assert!(tensor.narrow(1, u32::MAX, 2).is_err());
    }

    #[test]
    fn rejects_invalid_permutation_preconditions() {
        let tensor = Tensor::from_slice(&(0_u32..6).collect::<Vec<_>>(), &[2, 3]).unwrap();

        assert!(tensor.permute(&[0]).is_err());
        assert!(tensor.permute(&[0, 0]).is_err());
        assert!(tensor.permute(&[0, 2]).is_err());
    }

    #[test]
    fn rejects_transpose_below_rank_two() {
        let tensor = Tensor::from_slice(&[1_u32, 2], &[2]).unwrap();

        assert!(tensor.t().is_err());
    }

    #[test]
    fn rejects_invalid_allocation_preconditions() {
        assert!(Tensor::from_slice(&[1_u32], &[2]).is_err());
        assert!(Tensor::<u32>::from_slice(&[], &[u32::MAX, u32::MAX, u32::MAX]).is_err());
    }
}
