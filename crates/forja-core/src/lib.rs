//! Core types and contracts shared by Forja's trusted backends.
//! Use this crate to describe validated tensor views and operation signatures.

mod backend;
mod cost;
mod graph;
mod layout;
mod ops;
mod paged;
mod paged_pool;
mod profile_scope;
pub mod program;
mod quantized;
pub mod roofline;
mod symbolic;
mod variant_proof;
mod variants;

pub use backend::{
    AllocationRegistry, Backend, BackendError, DeviceInfo, DispatchProfile, KernelProfile,
    MemoryUsage, ProfileCount, ProfileTensor, ReadonlyImport, Submission, SubmissionProfile,
    ViewOp, automatic_buffer_pool_capacity, automatic_live_bytes,
};
pub use cost::{CostModel, KernelCost};
pub use forja_mmap::MappedRegion;
pub use graph::{
    GraphError, GraphLimits, GraphMemoryPlan, GraphTemplate, PreparedGraph, TemplateOp,
    TemplateTensor,
};
pub use layout::{Layout, LayoutError, MAX_RANK, Slice, byte_ranges_overlap, is_injective};
pub use ops::{
    BackendDispatchData, BufferId, CommandList, Dispatch, Op, OpError, Operand, PagedCopyDirection,
    Tensor, TensorError, gather_matmul_flops, gather_quant_matmul_dispatch_flops,
    gather_quant_matmul_flops, matmul_flops, quant_matmul_flops, required_barriers, sdpa_flops,
};
pub use paged::{
    MAX_PAGE_TOKENS, MAX_PAGED_PLANES, PageTable, PageTableEntry, PageTableError, PagedLayout,
    PagedLayoutError, PagedPlane,
};
pub use paged_pool::{
    CowPageCopy, PagePool, PagePoolError, PagedMemoryUsage, PagedSequence, PagedSlot,
    PagedSlotBanks, SequenceCheckpoint, paged_slab_bytes,
};
pub use profile_scope::{ProfileScope, ProfileScopeError};
pub use quantized::{QuantizedMatrix, QuantizedMatrixError, QuantizedMatrixPart};
pub use symbolic::{
    Affine, ByteHull, MAX_PARAMS, ParamError, ParamSpace, ParamValues, SymbolicLayout,
    SymbolicLayoutError,
};
pub use variant_proof::{ConstraintProofError, RuleArm, RuleError, VariantRule, prove_constraints};
pub use variants::{
    Axis, Constraint, DeviceCapability, Dimension, Guarantee, LayoutClass, Lifecycle,
    OperationKind, OperationValue, Relation, TensorRef, TensorSlot, ValueRef, VariantDef,
    operation_value, render_catalog_json, render_catalog_markdown,
};

/// Derives the deterministic counter stream used to correct a rejected speculative sample.
///
/// Speculate-then-correct keeps an allowed draw from the ordinary stream. A rejected draw is
/// replaced by an independently sampled draw from the masked distribution. Conditioning on the
/// allowed set therefore preserves its exact marginal distribution without reusing the rejected
/// draw's uniform value.
#[must_use]
pub const fn correction_sampling_seed(seed: u64) -> u64 {
    seed ^ (1_u64 << 63)
}

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
    use super::{DType, correction_sampling_seed};

    #[test]
    fn reports_scalar_byte_sizes() {
        assert_eq!(DType::F32.byte_size(), 4);
        assert_eq!(DType::F16.byte_size(), 2);
        assert_eq!(DType::BF16.byte_size(), 2);
        assert_eq!(DType::I32.byte_size(), 4);
        assert_eq!(DType::U32.byte_size(), 4);
    }

    #[test]
    fn correction_sampling_uses_a_distinct_deterministic_stream() {
        assert_eq!(correction_sampling_seed(17), correction_sampling_seed(17));
        assert_ne!(correction_sampling_seed(17), 17);
    }
}
