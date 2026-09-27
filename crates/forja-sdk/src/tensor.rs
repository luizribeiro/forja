use std::{marker::PhantomData, ops::Add, rc::Rc};

use crate::{Element, Error, Result, graph, sys};

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
///
/// Tensors are neither [`Send`] nor [`Sync`] because their lazy command graph
/// is local to the thread that created them.
///
/// ```compile_fail
/// use forja_sdk::Tensor;
/// fn require_send<T: Send>() {}
/// require_send::<Tensor<f32>>();
/// ```
///
/// ```compile_fail
/// use forja_sdk::Tensor;
/// fn require_sync<T: Sync>() {}
/// require_sync::<Tensor<f32>>();
/// ```
pub struct Tensor<T: Element> {
    handle: sys::Handle,
    shape: Vec<u32>,
    element: PhantomData<T>,
    not_thread_safe: PhantomData<Rc<()>>,
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

    /// Applies `SiLU` to this gate and multiplies it by `up`.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible tensors or a refused dispatch.
    pub fn silu_mul(&self, up: &Self) -> Result<Self> {
        self.binary(up, sys::Op::SiluMul)
    }

    /// Normalizes each row and applies a one-dimensional weight.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid epsilon, shape, type, or dispatch.
    pub fn rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        self.binary(weight, sys::Op::RmsNorm(eps))
    }

    /// Applies stable softmax over the last dimension.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid shape, type, or dispatch.
    pub fn softmax_last_dim(&self) -> Result<Self> {
        self.unary::<T>(sys::Op::Softmax, self.shape.clone())
    }

    /// Copies the tensor while converting between floating-point types.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported cast or refused dispatch.
    pub fn to_dtype<U: Element>(&self) -> Result<Tensor<U>> {
        self.unary(sys::Op::Copy, self.shape.clone())
    }

    /// Copies logical values into contiguous storage.
    ///
    /// # Errors
    ///
    /// Returns an error when the copy dispatch is refused.
    pub fn contiguous(&self) -> Result<Self> {
        self.unary(sys::Op::Copy, self.shape.clone())
    }

    /// Multiplies rank-two or rank-three matrices.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible ranks, shapes, types, or dispatches.
    pub fn matmul(&self, right: &Self) -> Result<Self> {
        let rank = self.shape.len();
        if !matches!(rank, 2 | 3) || right.shape.len() != rank {
            return Err(Error::new(
                "matmul requires equal rank-two or rank-three tensors",
            ));
        }
        let batch = rank - 2;
        if self.shape[..batch] != right.shape[..batch]
            || self.shape[batch + 1] != right.shape[batch]
        {
            return Err(Error::new("matmul shapes are incompatible"));
        }
        let mut shape = self.shape[..batch].to_vec();
        shape.extend([self.shape[batch], right.shape[batch + 1]]);
        self.binary_with_shape(right, sys::Op::Matmul, shape)
    }

    /// Applies half-split rotary position embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, theta, types, or dispatches.
    pub fn rope(&self, positions: &Tensor<u32>, theta: f32) -> Result<Self> {
        let output = Self::empty(self.shape.clone())?;
        graph::record(
            sys::Op::Rope(theta),
            &[&self.handle, &positions.handle],
            &output.handle,
        )?;
        Ok(output)
    }

    /// Gathers rows from this rank-two table by token id.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid shapes, ids, types, or dispatches.
    pub fn embedding(&self, ids: &Tensor<u32>) -> Result<Self> {
        let [_, width] = self
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::new("embedding table must have rank two"))?;
        let [count] = ids
            .shape
            .as_slice()
            .try_into()
            .map_err(|_| Error::new("embedding ids must have rank one"))?;
        let output = Self::empty(vec![count, width])?;
        graph::record(sys::Op::Embed, &[&self.handle, &ids.handle], &output.handle)?;
        Ok(output)
    }

    /// Records a copy into an existing writable destination view.
    ///
    /// # Errors
    ///
    /// Returns an error for incompatible shapes, types, aliasing, or dispatches.
    pub fn copy_into(&self, destination: &mut Self) -> Result<()> {
        graph::record(sys::Op::Copy, &[&self.handle], &destination.handle)
    }

    /// Submits pending work and gathers the logical tensor values.
    ///
    /// # Errors
    ///
    /// Returns an error when reading fails.
    pub fn to_vec(&self) -> Result<Vec<T>> {
        crate::eval()?;
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
            not_thread_safe: PhantomData,
        }
    }

    pub(crate) fn empty<U: Element>(shape: Vec<u32>) -> Result<Tensor<U>> {
        let handle = sys::alloc(U::DTYPE, &shape)?;
        Ok(Tensor::from_handle(handle, shape))
    }

    fn unary<U: Element>(&self, operation: sys::Op, shape: Vec<u32>) -> Result<Tensor<U>> {
        let output = Self::empty(shape)?;
        graph::record(operation, &[&self.handle], &output.handle)?;
        Ok(output)
    }

    fn binary(&self, other: &Self, operation: sys::Op) -> Result<Self> {
        self.binary_with_shape(other, operation, self.shape.clone())
    }

    fn binary_with_shape(&self, other: &Self, operation: sys::Op, shape: Vec<u32>) -> Result<Self> {
        let output = Self::empty(shape)?;
        graph::record(operation, &[&self.handle, &other.handle], &output.handle)?;
        Ok(output)
    }

    pub(crate) fn handle(&self) -> &sys::Handle {
        &self.handle
    }
}

impl<T: Element> Add<&Tensor<T>> for &Tensor<T> {
    type Output = Result<Tensor<T>>;

    fn add(self, other: &Tensor<T>) -> Self::Output {
        self.binary(other, sys::Op::Add)
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

    #[test]
    fn rejects_invalid_matmul_preconditions() {
        let vector = Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap();
        let matrix = Tensor::from_slice(&[1.0_f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let rank_three = Tensor::from_slice(&[1.0_f32; 8], &[2, 2, 2]).unwrap();
        let wrong_inner = Tensor::from_slice(&[1.0_f32; 6], &[3, 2]).unwrap();

        assert!(vector.matmul(&vector).is_err());
        assert!(matrix.matmul(&rank_three).is_err());
        assert!(matrix.matmul(&wrong_inner).is_err());
    }

    #[test]
    fn native_row_and_elementwise_ops_match_hand_values() {
        let left = Tensor::from_slice(&[1.0_f32, -2.0, 3.0], &[1, 3]).unwrap();
        let right = Tensor::from_slice(&[4.0_f32, 5.0, -1.0], &[1, 3]).unwrap();
        let sum = (&left + &right).unwrap();
        crate::eval().unwrap();
        assert_close(&sum.to_vec().unwrap(), &[5.0, 3.0, 2.0]);

        let gate = Tensor::from_slice(&[0.0_f32, 1.0, -1.0], &[1, 3]).unwrap();
        let up = Tensor::from_slice(&[2.0_f32, 3.0, 4.0], &[1, 3]).unwrap();
        let activated = gate.silu_mul(&up).unwrap();
        let expected = [
            0.0,
            3.0 / (1.0 + (-1.0_f32).exp()),
            -4.0 / (1.0 + 1.0_f32.exp()),
        ];
        assert_close(&activated.to_vec().unwrap(), &expected);

        let input = Tensor::from_slice(&[3.0_f32, 4.0], &[1, 2]).unwrap();
        let weight = Tensor::from_slice(&[1.0_f32, 2.0], &[2]).unwrap();
        let normalized = input.rms_norm(&weight, 0.0).unwrap();
        let rms = 12.5_f32.sqrt();
        assert_close(&normalized.to_vec().unwrap(), &[3.0 / rms, 8.0 / rms]);

        let logits = Tensor::from_slice(&[0.0_f32, 2.0_f32.ln()], &[1, 2]).unwrap();
        let probabilities = logits.softmax_last_dim().unwrap();
        assert_close(&probabilities.to_vec().unwrap(), &[1.0 / 3.0, 2.0 / 3.0]);

        let half = sum.to_dtype::<crate::f16>().unwrap();
        assert_eq!(
            half.to_vec()
                .unwrap()
                .into_iter()
                .map(crate::f16::to_f32)
                .collect::<Vec<_>>(),
            [5.0, 3.0, 2.0]
        );
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        let error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).powi(2))
            .sum::<f32>()
            .sqrt();
        let norm = expected
            .iter()
            .map(|value| value.powi(2))
            .sum::<f32>()
            .sqrt();
        assert!(
            error <= 1.0e-5 * norm.max(1.0),
            "{actual:?} != {expected:?}"
        );
    }
}
