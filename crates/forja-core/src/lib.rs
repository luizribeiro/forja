//! Core types and contracts shared by Forja's trusted backends.
//! Use this crate to describe validated tensor views and operation signatures.

mod backend;
mod graph;
mod layout;
mod ops;
pub mod program;
mod quantized;
mod symbolic;

pub use backend::{
    AllocationRegistry, Backend, BackendError, DispatchProfile, ProfileCount, ProfileTensor,
    Submission, SubmissionProfile, ViewOp,
};
pub use forja_mmap::MappedRegion;
pub use graph::{
    GraphError, GraphLimits, GraphTemplate, PreparedGraph, TemplateOp, TemplateTensor,
};
pub use layout::{Layout, LayoutError, MAX_RANK, Slice, byte_ranges_overlap, is_injective};
pub use ops::{
    BufferId, CommandList, Dispatch, Op, OpError, Operand, Tensor, TensorError,
    gather_quant_matmul_flops, matmul_flops, quant_matmul_flops, required_barriers, sdpa_flops,
};
pub use quantized::{QuantizedMatrix, QuantizedMatrixError, QuantizedMatrixPart};
pub use symbolic::{
    Affine, ByteHull, MAX_PARAMS, ParamError, ParamSpace, ParamValues, SymbolicLayout,
    SymbolicLayoutError,
};

/// A scalar type stored in an unquantized tensor.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum DType {
    /// A 32-bit IEEE 754 floating-point value.
    F32,
    /// A 16-bit IEEE 754 floating-point value.
    F16,
    /// A 16-bit brain floating-point value.
    BF16,
    /// A signed 32-bit integer.
    I32,
    /// An unsigned 32-bit integer.
    U32,
}

impl DType {
    /// Returns the number of bytes occupied by one value of this type.
    #[must_use]
    pub const fn byte_size(self) -> u64 {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DType;

    #[test]
    fn reports_scalar_byte_sizes() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::BF16.byte_size(), 2);
        assert_eq!(DType::I32.byte_size(), 4);
        assert_eq!(DType::U32.byte_size(), 4);
    }
}
