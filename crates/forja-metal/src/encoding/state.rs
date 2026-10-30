use forja_core::{StateOp, Tensor};

use super::{
    ArgumentBindings, ArgumentWriter, BackendError, BufferBinding, Dispatch, MTL4ArgumentTable,
    MTLComputePipelineState, MTLSize, MetalBackend, Op, ProtocolObject, dispatch_threadgroups,
    dtype_code, encode_layout, error_state, set_argument_table, set_pipeline,
};

impl MetalBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_state(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<(Vec<BufferBinding>, BufferBinding), BackendError> {
        let Op::State(operation) = dispatch.op() else {
            return Err(BackendError::InvalidInput);
        };
        let input = dispatch.inputs();
        let ordinary = input.len() - 3;
        let (kernel, geometry, groups) = state_geometry(operation, input, dispatch.outputs())?;
        let tensors = input
            .iter()
            .chain(dispatch.outputs())
            .map(|t| self.encoder_tensor(t))
            .collect::<Result<Vec<_>, _>>()?;
        let mut layout_bytes = Vec::new();
        for tensor in tensors[..ordinary].iter().chain(&tensors[input.len()..]) {
            layout_bytes.extend(encode_layout(&tensor.layout)?);
        }
        let layouts = arguments.write(&layout_bytes)?;
        let access = operation.access();
        let params: Vec<_> = [
            access.source(),
            access.destination(),
            input[input.len() - 2].layout().shape()[0],
        ]
        .into_iter()
        .chain(geometry)
        .flat_map(u32::to_ne_bytes)
        .collect();
        let params = arguments.write(&params)?;
        let flag = arguments.write(&error_state())?;
        let constants: Vec<_> = [0, 1, 3, 4, 5]
            .into_iter()
            .enumerate()
            .take(ordinary)
            .map(|(i, slot)| (slot, dtype_code(input[i].layout().dtype())))
            .chain([(2, dtype_code(dispatch.output().layout().dtype()))])
            .collect();
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, &constants)?;
        if pipeline.maxTotalThreadsPerThreadgroup() < 128 {
            return Err(BackendError::UnsupportedOperation);
        }
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in tensors.iter().enumerate() {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, tensors.len(), &layouts);
        bindings.bind(table, tensors.len() + 1, &params);
        bindings.bind(table, tensors.len() + 2, &flag);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            groups,
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((vec![layouts, params, flag.clone()], flag))
    }
}

fn state_geometry(
    operation: StateOp,
    input: &[Tensor],
    outputs: &[Tensor],
) -> Result<(&'static str, [u32; 4], MTLSize), BackendError> {
    let mut groups = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    match operation {
        StateOp::DeltaRuleBlock(_) => Err(BackendError::UnsupportedOperation),
        StateOp::DeltaRule(_) => {
            let s = input[5].layout().shape();
            if u64::from(s[1]) > crate::geometry::DELTA_RULE_MAX_KEYS {
                return Err(BackendError::UnsupportedOperation);
            }
            groups.width =
                usize::try_from(s[2].div_ceil(4)).map_err(|_| BackendError::InvalidInput)?;
            groups.height = usize::try_from(s[0]).map_err(|_| BackendError::InvalidInput)?;
            Ok((
                "state_delta_rule",
                [input[0].layout().shape()[0], s[0], s[1], s[2]],
                groups,
            ))
        }
        StateOp::CausalDepthwiseConv(_) => {
            let s = input[0].layout().shape();
            let count = outputs
                .iter()
                .map(|t| t.layout().element_count())
                .max()
                .ok_or(BackendError::InvalidInput)?;
            groups.width =
                usize::try_from(count.div_ceil(128)).map_err(|_| BackendError::InvalidInput)?;
            Ok((
                "state_causal_conv",
                [s[0], s[1], input[1].layout().shape()[0], 0],
                groups,
            ))
        }
    }
}
