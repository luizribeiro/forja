use super::{
    BackendError, Dispatch, MTLComputePipelineState, MetalBackend, Op, dtype_code,
    epilogue_codegen, qmv_vector_config, quant_epilogue_specialization, quant_matmul_kernel,
};

/// Compiled pipeline limits, without inferring register allocation or occupancy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelResources {
    /// Specialized shader entry point.
    pub kernel: &'static str,
    /// Hardware execution width reported by the compiled pipeline.
    pub execution_width: u64,
    /// Maximum legal total threadgroup threads for this pipeline.
    pub maximum_threads: u64,
    /// Statically allocated threadgroup storage in bytes.
    pub static_threadgroup_bytes: u64,
}

impl MetalBackend {
    /// Reports compiled resources for supported diagnostic operation families.
    /// Call after execution so a fused epilogue's pipeline is already prepared.
    ///
    /// # Errors
    /// Refuses invalid dimensions, missing epilogue preparation or compilation failure.
    pub fn kernel_resources(
        &self,
        dispatch: &Dispatch,
    ) -> Result<Option<KernelResources>, BackendError> {
        if !matches!(dispatch.op(), Op::QuantMatmul { .. }) {
            return Ok(None);
        }
        let kernel = quant_matmul_kernel(dispatch)?;
        let constants = quant_constants(dispatch)?;
        let pipelines = &self.pipelines;
        let pipeline = if let Some(epilogue) = dispatch.prepared_epilogue() {
            pipelines.lookup_epilogue(
                epilogue_codegen::FamilySourceTag::Quant,
                epilogue,
                kernel,
                &constants,
            )?
        } else {
            pipelines.get(kernel, &constants)?
        };
        Ok(Some(KernelResources {
            kernel,
            execution_width: u64::try_from(pipeline.threadExecutionWidth())
                .map_err(|_| BackendError::InvalidInput)?,
            maximum_threads: u64::try_from(pipeline.maxTotalThreadsPerThreadgroup())
                .map_err(|_| BackendError::InvalidInput)?,
            static_threadgroup_bytes: u64::try_from(pipeline.staticThreadgroupMemoryLength())
                .map_err(|_| BackendError::InvalidInput)?,
        }))
    }
}

pub(super) fn quant_constants(dispatch: &Dispatch) -> Result<[(u32, u32); 10], BackendError> {
    let Op::QuantMatmul { bits, group_size } = dispatch.op() else {
        return Err(BackendError::UnsupportedOperation);
    };
    let [input, _, scales, _] = dispatch.primitive_inputs() else {
        return Err(BackendError::InvalidInput);
    };
    let inner = input
        .layout()
        .shape()
        .last()
        .copied()
        .ok_or(BackendError::InvalidInput)?;
    let columns = dispatch
        .output()
        .layout()
        .shape()
        .last()
        .copied()
        .ok_or(BackendError::InvalidInput)?;
    let vector = qmv_vector_config(dispatch);
    Ok([
        (0, dtype_code(input.layout().dtype())),
        (1, dtype_code(scales.layout().dtype())),
        (2, dtype_code(dispatch.output().layout().dtype())),
        (12, u32::from(bits)),
        (13, group_size),
        (222, vector[0]),
        (223, vector[1]),
        (224, vector[2]),
        (225, vector[3]),
        (15, quant_epilogue_specialization(columns, inner)),
    ])
}
