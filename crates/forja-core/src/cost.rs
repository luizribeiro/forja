//! Logical algorithm work; cache reuse and instruction throughput are measured separately.

mod attention;
#[cfg(test)]
mod copies;
mod routed;
pub use routed::quantized_route_index;
mod scalar;
mod tiling;
pub use tiling::CostModel;

use crate::{Dispatch, Op, Tensor, matmul_flops, quant_matmul_flops};

/// Checked logical traffic and arithmetic for a validated operation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelCost {
    /// Logical operand bytes read, including quantization metadata.
    pub read_bytes: u64,
    /// Logical result bytes written.
    pub write_bytes: u64,
    /// Multiply and add operations, counting an FMA as two operations.
    pub flops: u64,
    /// Scalar arithmetic, comparisons and special functions (one operation each).
    pub scalar_ops: u64,
    /// Weight bytes included in reads, before data-dependent expert reuse.
    pub weight_bytes: u64,
    /// Minimum footprint for distinct per-row routes; zero for non-routed operations.
    /// Profilers must lower this bound when route uniqueness is unproved.
    pub minimum_weight_bytes: u64,
    /// Largest distinct routed weight footprint allowed by the shape.
    pub maximum_weight_bytes: u64,
    /// Logical K/V bytes included in reads.
    pub kv_bytes: u64,
}

impl KernelCost {
    /// Returns a machine-readable declaration with explicit units.
    #[must_use]
    pub fn json(self) -> serde_json::Value {
        serde_json::json!({"read_bytes": self.read_bytes, "write_bytes": self.write_bytes,
            "flops": self.flops, "scalar_ops": self.scalar_ops,
            "weight_bytes": self.weight_bytes, "kv_bytes": self.kv_bytes,
            "minimum_weight_bytes": self.minimum_weight_bytes,
            "maximum_weight_bytes": self.maximum_weight_bytes})
    }

    /// Counts a validated dispatch, refusing any overflow.
    #[must_use]
    pub fn of(dispatch: &Dispatch) -> Option<Self> {
        let read_bytes = bytes(dispatch.inputs())?;
        let write_bytes = bytes(dispatch.outputs())?;
        let mut cost = Self {
            read_bytes,
            write_bytes,
            ..Self::default()
        };
        let inputs = dispatch.primitive_inputs();
        match dispatch.op() {
            Op::Matmul => {
                cost.flops = matmul_flops(inputs[0].layout().shape(), inputs[1].layout().shape())?;
                cost.weight_bytes = tensor_bytes(&inputs[1])?;
            }
            Op::QuantMatmul { .. } => {
                cost.flops =
                    quant_matmul_flops(inputs[0].layout().shape(), inputs[1].layout().shape())?;
                cost.weight_bytes = bytes(&inputs[1..4])?;
            }
            _ => {}
        }
        scalar::count(
            dispatch,
            routed::count(dispatch, attention::count(dispatch, cost)?)?,
        )
    }
}

pub(super) fn bytes(tensors: &[Tensor]) -> Option<u64> {
    tensors
        .iter()
        .try_fold(0_u64, |sum, tensor| sum.checked_add(tensor_bytes(tensor)?))
}

pub(super) fn tensor_bytes(tensor: &Tensor) -> Option<u64> {
    product(
        &tensor
            .layout()
            .shape()
            .iter()
            .copied()
            .map(u64::from)
            .collect::<Vec<_>>(),
    )?
    .checked_mul(tensor.layout().dtype().byte_size())
}

pub(super) fn product(values: &[u64]) -> Option<u64> {
    values
        .iter()
        .try_fold(1_u64, |product, value| product.checked_mul(*value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferId, CommandList, DType, Layout};

    pub(super) fn tensor(id: u64, dtype: DType, shape: &[u32]) -> Tensor {
        let size = product(&shape.iter().copied().map(u64::from).collect::<Vec<_>>()).unwrap()
            * dtype.byte_size();
        Tensor::from_allocation(
            BufferId::new(1, id, size),
            Layout::contiguous(dtype, 0, shape.to_vec(), size).unwrap(),
            true,
        )
        .unwrap()
    }

    #[test]
    fn dense_gemv_and_gemm_count_logical_work() {
        for rows in [1, 7, 33] {
            let a = tensor(1, DType::BF16, &[rows, 1024]);
            let b = tensor(2, DType::BF16, &[1024, 4097]);
            let c = tensor(3, DType::BF16, &[rows, 4097]);
            let mut commands = CommandList::new();
            commands.dispatch(Op::Matmul, &[&a, &b], &c).unwrap();
            let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
            assert_eq!(cost.flops, 2 * u64::from(rows) * 1024 * 4097);
            assert_eq!(cost.read_bytes, 2 * (u64::from(rows) * 1024 + 1024 * 4097));
            assert_eq!(cost.write_bytes, 2 * u64::from(rows) * 4097);
        }
    }

    #[test]
    fn quantized_weights_include_scales_and_biases() {
        let input = tensor(1, DType::BF16, &[7, 1024]);
        let packed = tensor(2, DType::U32, &[33, 128]);
        let scales = tensor(3, DType::BF16, &[33, 16]);
        let biases = tensor(4, DType::BF16, &[33, 16]);
        let output = tensor(5, DType::BF16, &[7, 33]);
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::QuantMatmul {
                    bits: 4,
                    group_size: 64,
                },
                &[&input, &packed, &scales, &biases],
                &output,
            )
            .unwrap();
        let cost = KernelCost::of(commands.last_dispatch().unwrap()).unwrap();
        assert_eq!(cost.flops, 2 * 7 * 1024 * 33);
        assert_eq!(cost.weight_bytes, 33 * (128 * 4 + 16 * 4));
        assert_eq!(product(&[u64::MAX, 2]), None);
    }
}
