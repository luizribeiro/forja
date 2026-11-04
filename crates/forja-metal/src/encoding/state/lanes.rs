use super::{
    ArgumentBindings, ArgumentWriter, BackendError, BufferBinding, Dispatch, MTL4ArgumentTable,
    MTLComputePipelineState, MTLSize, MetalBackend, Op, ProtocolObject, Tensor,
    dispatch_threadgroups, dtype_code, encode_layout, error_state, set_argument_table,
    set_pipeline, state_geometry,
};
use crate::variants::MetalVariant;

pub(in crate::encoding) fn lane_variant(dispatch: &Dispatch) -> bool {
    matches!(
        super::super::selected_variant(dispatch),
        Some(MetalVariant::DeltaRuleLanes | MetalVariant::CausalDepthwiseConvLanes)
    )
}

pub(in crate::encoding) fn lane_group(dispatches: &[Dispatch]) -> usize {
    let Some(first) = dispatches.first().filter(|first| lane_variant(first)) else {
        return 1;
    };
    let mut count = 1;
    for (index, candidate) in dispatches.iter().enumerate().take(8).skip(1) {
        if super::super::selected_variant(candidate) != super::super::selected_variant(first)
            || !first
                .inputs()
                .iter()
                .chain(first.outputs())
                .zip(candidate.inputs().iter().chain(candidate.outputs()))
                .all(|(a, b)| {
                    a.layout().shape() == b.layout().shape()
                        && a.layout().dtype() == b.layout().dtype()
                })
            || dispatches[..index]
                .iter()
                .any(|prior| !independent(prior, candidate))
        {
            break;
        }
        count += 1;
    }
    count
}

fn independent(left: &Dispatch, right: &Dispatch) -> bool {
    let conflicts = |outputs: &[Tensor], other: &Dispatch| {
        outputs.iter().any(|output| {
            other.inputs().iter().chain(other.outputs()).any(|input| {
                output.buffer() == input.buffer()
                    && forja_core::byte_ranges_overlap(output.layout(), input.layout())
            })
        })
    };
    !conflicts(left.outputs(), right) && !conflicts(right.outputs(), left)
}

impl MetalBackend {
    pub(in crate::encoding) fn encode_state_lanes(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatches: &[Dispatch],
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
    ) -> Result<(Vec<BufferBinding>, BufferBinding), BackendError> {
        let first = &dispatches[0];
        let Op::State(operation) = first.op() else {
            return Err(BackendError::InvalidInput);
        };
        let (kernel, _, mut groups) = state_geometry(operation, first.inputs(), first.outputs())?;
        groups.depth = dispatches.len();
        let kernel = match kernel {
            "state_delta_rule" => "state_delta_rule_lanes",
            "state_causal_conv" => "state_causal_conv_lanes",
            _ => return Err(BackendError::UnsupportedOperation),
        };
        let mut pointers = Vec::new();
        let mut layouts = Vec::new();
        let mut params = Vec::new();
        for dispatch in dispatches {
            let Op::State(operation) = dispatch.op() else {
                return Err(BackendError::InvalidInput);
            };
            let input = dispatch.inputs();
            let ordinary = input.len() - 3;
            let (_, geometry, _) = state_geometry(operation, input, dispatch.outputs())?;
            let tensors = input
                .iter()
                .chain(dispatch.outputs())
                .map(|t| self.encoder_tensor(t))
                .collect::<Result<Vec<_>, _>>()?;
            for tensor in &tensors {
                pointers.extend(tensor.buffer.address.to_ne_bytes());
            }
            pointers.resize(pointers.len() + (11 - tensors.len()) * 8, 0);
            for tensor in tensors[..ordinary].iter().chain(&tensors[input.len()..]) {
                layouts.extend(encode_layout(&tensor.layout)?);
            }
            layouts.resize(
                layouts.len() + (8 - ordinary - dispatch.outputs().len()) * 112,
                0,
            );
            let access = operation.access();
            for value in [
                access.source(),
                access.destination(),
                input[input.len() - 2].layout().shape()[0],
                u32::from(access.is_active()),
            ]
            .into_iter()
            .chain(geometry)
            {
                params.extend(value.to_ne_bytes());
            }
        }
        let constants = [0, 1, 3, 4, 5]
            .into_iter()
            .enumerate()
            .take(first.inputs().len() - 3)
            .map(|(i, slot)| (slot, dtype_code(first.inputs()[i].layout().dtype())))
            .chain([(2, dtype_code(first.output().layout().dtype()))])
            .collect::<Vec<_>>();
        let pipeline = self
            .pipelines
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(kernel, &constants)?;
        if pipeline.maxTotalThreadsPerThreadgroup() < 128 {
            return Err(BackendError::UnsupportedOperation);
        }
        let buffers = [pointers, layouts, params, error_state().to_vec()]
            .iter()
            .map(|bytes| arguments.write(bytes))
            .collect::<Result<Vec<_>, _>>()?;
        set_pipeline(encoder, &pipeline);
        for (index, buffer) in buffers.iter().enumerate() {
            bindings.bind(table, index, buffer);
        }
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
        let flag = buffers[3].clone();
        Ok((buffers, flag))
    }
}

pub(in crate::encoding) fn fold_profiles(
    dispatches: &[Dispatch],
    profiles: &mut [forja_core::DispatchProfile],
) -> Result<(), BackendError> {
    let mut index = 0;
    while index < dispatches.len() {
        let count = lane_group(&dispatches[index..]);
        for member in index + 1..index + count {
            profiles[index].cost = profiles[index]
                .cost
                .checked_add(profiles[member].cost)
                .ok_or(BackendError::InvalidInput)?;
            profiles[member].cost = forja_core::KernelCost::default();
            for bound in 0..2 {
                profiles[index].padded_flops[bound] = profiles[index].padded_flops[bound]
                    .checked_add(profiles[member].padded_flops[bound])
                    .ok_or(BackendError::InvalidInput)?;
            }
            profiles[member].padded_flops = [0; 2];
        }
        index += count;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{BackendDispatchData, CommandList, DType, DeviceCapability, StateOp};

    fn convolution(backend: &forja_cpu::CpuBackend, channels: u32) -> Dispatch {
        let (access, _) = forja_testing::state_test_metadata().unwrap();
        let inputs = [
            (DType::F32, vec![1, channels]),
            (DType::F32, vec![4, channels]),
            (DType::F32, vec![3, channels]),
            (DType::U32, vec![2, 2]),
            (DType::U32, vec![3]),
            (DType::U32, vec![1]),
        ]
        .map(|(dtype, shape)| backend.alloc(dtype, &shape).unwrap());
        let output = backend.alloc(DType::F32, &[1, channels]).unwrap();
        let state = backend.alloc(DType::F32, &[3, channels]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch_many(
                Op::State(StateOp::CausalDepthwiseConv(access)),
                &inputs.iter().collect::<Vec<_>>(),
                &[&output, &state],
            )
            .unwrap();
        let variant = crate::variants::resolve(
            commands.last_dispatch().unwrap(),
            "causal-depthwise-conv.direct-lanes",
            &[DeviceCapability::Metal4],
        )
        .unwrap();
        commands.set_last_backend_data(BackendDispatchData::new(variant));
        commands.into_dispatches().remove(0)
    }

    #[test]
    fn lane_groups_require_independent_matching_views_and_cap_the_record_table() {
        let backend = forja_cpu::CpuBackend::new();
        let first = convolution(&backend, 33);
        let second = convolution(&backend, 33);
        assert_eq!(lane_group(&[first.clone(), second]), 2);
        assert_eq!(lane_group(&[first.clone(), first.clone()]), 1);
        assert_eq!(lane_group(&[first, convolution(&backend, 7)]), 1);
        let lanes = (0..9)
            .map(|_| convolution(&backend, 33))
            .collect::<Vec<_>>();
        assert_eq!(lane_group(&lanes), 8);
    }
}
