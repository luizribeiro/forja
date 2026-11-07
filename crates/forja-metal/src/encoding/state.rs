use forja_core::{StateOp, Tensor};

use super::{
    ArgumentBindings, ArgumentWriter, BackendError, BufferBinding, Dispatch, EncoderTensor,
    MTL4ArgumentTable, MTLComputePipelineState, MTLSize, MetalBackend, Op, ProtocolObject,
    Retained, dispatch_threadgroups, dtype_code, encode_layout, error_state, set_argument_table,
    set_pipeline,
};

mod block;
mod lanes;
mod steps;
pub(super) use lanes::{fold_profiles, lane_group, lane_variant, steps_variant};
#[cfg(test)]
mod block_tests;

impl MetalBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_state(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
        timestamps: Option<&super::GpuTimestamps>,
        internal_timestamp: &mut usize,
    ) -> Result<(Vec<BufferBinding>, BufferBinding), BackendError> {
        let Op::State(operation) = dispatch.op() else {
            return Err(BackendError::InvalidInput);
        };
        if matches!(operation, StateOp::DeltaRuleBlock(_)) {
            return self.encode_delta_block(
                encoder,
                table,
                dispatch,
                bindings,
                arguments,
                timestamps,
                internal_timestamp,
            );
        }
        if lane_variant(dispatch) {
            return self.encode_state_lanes(
                encoder,
                table,
                std::slice::from_ref(dispatch),
                bindings,
                arguments,
            );
        }
        let input = dispatch.inputs();
        let (kernel, geometry, groups) = state_geometry(operation, input, dispatch.outputs())?;
        let tensors = input
            .iter()
            .chain(dispatch.outputs())
            .map(|t| self.encoder_tensor(t))
            .collect::<Result<Vec<_>, _>>()?;
        let (layout_bytes, parameter_bytes) = state_arguments(dispatch, &tensors, geometry)?;
        let layouts = arguments.write(&layout_bytes)?;
        let params = arguments.write(&parameter_bytes)?;
        let flag = arguments.write(&error_state())?;
        let pipeline = self.state_pipeline(kernel, std::slice::from_ref(dispatch))?;
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
    fn state_pipeline(
        &self,
        kernel: &str,
        dispatches: &[Dispatch],
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BackendError> {
        let dispatch = dispatches.first().ok_or(BackendError::InvalidInput)?;
        let input = dispatch.inputs();
        let linear = matches!(
            super::selected_variant(dispatch),
            Some(crate::variants::MetalVariant::DeltaRuleLinearLanes)
        );
        let linear_views = linear_view_mask(dispatches)?;
        let ordinary = input
            .len()
            .checked_sub(3)
            .ok_or(BackendError::InvalidInput)?;
        let constants = [0, 1, 3, 4, 5]
            .into_iter()
            .enumerate()
            .take(ordinary)
            .map(|(i, slot)| (slot, dtype_code(input[i].layout().dtype())))
            .chain([
                (2, dtype_code(dispatch.output().layout().dtype())),
                (227, u32::from(linear)),
                (228, linear_views),
            ])
            .collect::<Vec<_>>();
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, &constants)?;
        if pipeline.maxTotalThreadsPerThreadgroup() < 128 {
            return Err(BackendError::UnsupportedOperation);
        }
        Ok(pipeline)
    }
}

fn linear_view_mask(dispatches: &[Dispatch]) -> Result<u32, BackendError> {
    let first = dispatches.first().ok_or(BackendError::InvalidInput)?;
    if super::selected_variant(first) != Some(crate::variants::MetalVariant::DeltaRuleLinearLanes) {
        return Ok(0);
    }
    dispatches.iter().try_fold(u32::MAX, |mask, dispatch| {
        Ok(mask & contiguous_views(dispatch)?)
    })
}

fn contiguous_views(dispatch: &Dispatch) -> Result<u32, BackendError> {
    let inputs = dispatch.inputs();
    let ordinary = inputs
        .len()
        .checked_sub(3)
        .ok_or(BackendError::InvalidInput)?;
    inputs
        .get(..ordinary)
        .ok_or(BackendError::InvalidInput)?
        .iter()
        .chain(dispatch.outputs())
        .enumerate()
        .try_fold(0_u32, |mask, (index, tensor)| {
            let bit = 1_u32
                .checked_shl(u32::try_from(index).map_err(|_| BackendError::InvalidInput)?)
                .ok_or(BackendError::InvalidInput)?;
            Ok(mask
                | if tensor.layout().is_contiguous() {
                    bit
                } else {
                    0
                })
        })
}

fn state_arguments(
    dispatch: &Dispatch,
    tensors: &[EncoderTensor],
    geometry: [u32; 4],
) -> Result<(Vec<u8>, Vec<u8>), BackendError> {
    let Op::State(operation) = dispatch.op() else {
        return Err(BackendError::InvalidInput);
    };
    let input = dispatch.inputs();
    let ordinary = input
        .len()
        .checked_sub(3)
        .ok_or(BackendError::InvalidInput)?;
    let mut layouts = Vec::new();
    for tensor in tensors
        .get(..ordinary)
        .ok_or(BackendError::InvalidInput)?
        .iter()
        .chain(&tensors[input.len()..])
    {
        layouts.extend(encode_layout(&tensor.layout)?);
    }
    let access = operation.access();
    let params = [
        access.source(),
        access.destination(),
        input[input.len() - 2].layout().shape()[0],
        u32::from(access.is_active()),
    ]
    .into_iter()
    .chain(geometry)
    .flat_map(u32::to_ne_bytes)
    .collect();
    Ok((layouts, params))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_masks_refuse_missing_metadata_and_ignore_nonlinear_layouts() {
        use forja_core::{CommandList, DType};
        let cpu = forja_cpu::CpuBackend::new();
        let input = cpu.alloc(DType::F32, &[7]).unwrap();
        let output = cpu.alloc(DType::F32, &[7]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::RmsNorm { eps: 1.0e-6 }, &[&input, &input], &output)
            .unwrap();
        let dispatch = commands.last_dispatch().unwrap();
        assert_eq!(contiguous_views(dispatch), Err(BackendError::InvalidInput));
        assert_eq!(linear_view_mask(commands.dispatches()), Ok(0));
        assert_eq!(linear_view_mask(&[]), Err(BackendError::InvalidInput));
    }

    #[test]
    fn metal_compiles_delta_block_stages_with_bounded_groups() {
        let backend = MetalBackend::new().unwrap();
        let mut pipelines = backend.pipelines.lock().unwrap();
        for name in [
            "state_block_gram",
            "state_block_gram_mma",
            "state_block_solve",
            "state_block_solve_panel",
            "state_block_carry",
            "state_block_carry_mma",
            "state_block_output",
            "state_block_output_mma",
        ] {
            let pipeline = pipelines
                .get(name, &[(0, 0), (1, 0), (2, 2), (3, 2), (4, 0), (5, 2)])
                .unwrap();
            assert!(pipeline.maxTotalThreadsPerThreadgroup() >= 128);
            assert!(pipeline.staticThreadgroupMemoryLength() <= 32 * 1024);
        }
    }
}
