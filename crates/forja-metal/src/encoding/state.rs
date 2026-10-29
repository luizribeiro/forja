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
        let Op::State(operation @ forja_core::StateOp::DeltaRule(_)) = dispatch.op() else {
            return Err(BackendError::UnsupportedOperation);
        };
        let input = dispatch.inputs();
        let state = input[5].layout().shape();
        if state[1] > 256 {
            return Err(BackendError::UnsupportedOperation);
        }
        let tensors = input
            .iter()
            .chain(dispatch.outputs())
            .map(|t| self.encoder_tensor(t))
            .collect::<Result<Vec<_>, _>>()?;
        let mut layout_bytes = Vec::new();
        for tensor in tensors[..6].iter().chain(&tensors[9..]) {
            layout_bytes.extend(encode_layout(&tensor.layout)?);
        }
        let layouts = arguments.write(&layout_bytes)?;
        let access = operation.access();
        let params: Vec<_> = [
            access.source(),
            access.destination(),
            input[7].layout().shape()[0],
            input[0].layout().shape()[0],
            state[0],
            state[1],
            state[2],
        ]
        .into_iter()
        .flat_map(u32::to_ne_bytes)
        .collect();
        let params = arguments.write(&params)?;
        let flag = arguments.write(&error_state())?;
        let constants: Vec<_> = [0, 1, 3, 4, 5]
            .into_iter()
            .enumerate()
            .map(|(i, slot)| (slot, dtype_code(input[i].layout().dtype())))
            .chain([(2, dtype_code(dispatch.output().layout().dtype()))])
            .collect();
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get("state_delta_rule", &constants)?;
        if pipeline.maxTotalThreadsPerThreadgroup() < 128 {
            return Err(BackendError::UnsupportedOperation);
        }
        set_pipeline(encoder, &pipeline);
        for (index, tensor) in tensors.iter().enumerate() {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, 11, &layouts);
        bindings.bind(table, 12, &params);
        bindings.bind(table, 13, &flag);
        set_argument_table(encoder, table);
        dispatch_threadgroups(
            encoder,
            MTLSize {
                width: usize::try_from(state[2].div_ceil(4))
                    .map_err(|_| BackendError::InvalidInput)?,
                height: usize::try_from(state[0]).map_err(|_| BackendError::InvalidInput)?,
                depth: 1,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok((vec![layouts, params, flag.clone()], flag))
    }
}
