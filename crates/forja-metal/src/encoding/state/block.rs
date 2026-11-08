use forja_core::DeltaBlockScratch;
use objc2_metal::MTLComputePipelineState;

use super::{
    ArgumentBindings, ArgumentWriter, BackendError, BufferBinding, Dispatch, MTL4ArgumentTable,
    MTLSize, MetalBackend, Op, ProtocolObject, dispatch_threadgroups, dtype_code, encode_layout,
    error_state, set_argument_table,
};

#[cfg(test)]
pub(super) struct StageTest {
    pub(super) stage: usize,
    pub(super) kernel: &'static str,
    pub(super) panel: Option<u32>,
}

fn selected_panel(dispatch: &Dispatch) -> Option<u32> {
    #[cfg(test)]
    if let Some(test) = dispatch.backend_data::<StageTest>() {
        return test.panel;
    }
    super::super::delta_block_mma_panel(dispatch)
}

pub(crate) fn block_schedule(dispatch: &Dispatch) -> Option<(Option<u32>, [u64; 4])> {
    let inputs = dispatch.primitive_inputs();
    let query = inputs.first()?.layout().shape();
    let state = inputs.get(5)?.layout().shape();
    let scratch = DeltaBlockScratch::new(
        *query.first()?,
        *state.first()?,
        *state.get(1)?,
        *state.get(2)?,
    )
    .ok()?;
    let panel = selected_panel(dispatch);
    let mut groups = [0_u64; 4];
    for (stage, count) in groups.iter_mut().enumerate() {
        #[cfg(test)]
        if dispatch
            .backend_data::<StageTest>()
            .is_some_and(|test| test.stage != stage)
        {
            continue;
        }
        let (grid, _) = stage_geometry(
            stage,
            scratch,
            state,
            dispatch.output().layout().element_count(),
            panel,
        )
        .ok()?;
        *count = u64::try_from(grid.width)
            .ok()?
            .checked_mul(u64::try_from(grid.height).ok()?)?
            .checked_mul(u64::try_from(grid.depth).ok()?)?;
    }
    Some((panel, groups))
}

impl MetalBackend {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_delta_block(
        &self,
        encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
        table: &ProtocolObject<dyn MTL4ArgumentTable>,
        dispatch: &Dispatch,
        bindings: &mut ArgumentBindings,
        arguments: &mut ArgumentWriter,
        timestamps: Option<&super::super::GpuTimestamps>,
        internal_timestamp: &mut usize,
    ) -> Result<(Vec<BufferBinding>, BufferBinding), BackendError> {
        let input = dispatch.inputs();
        let query = input[0].layout().shape();
        let state = input[5].layout().shape();
        if u64::from(state[1]) > crate::geometry::DELTA_RULE_MAX_KEYS {
            return Err(BackendError::UnsupportedOperation);
        }
        let scratch = DeltaBlockScratch::new(query[0], state[0], state[1], state[2])
            .map_err(|_| BackendError::InvalidInput)?;
        let panel = selected_panel(dispatch);
        #[cfg(test)]
        let test = dispatch.backend_data::<StageTest>();
        let tensors = input
            .iter()
            .chain(dispatch.outputs())
            .map(|tensor| self.encoder_tensor(tensor))
            .collect::<Result<Vec<_>, _>>()?;
        let mut layouts = Vec::new();
        for tensor in tensors[..6].iter().chain(&tensors[input.len()..]) {
            layouts.extend(encode_layout(&tensor.layout)?);
        }
        let layouts = arguments.write(&layouts)?;
        let Op::State(operation) = dispatch.op() else {
            return Err(BackendError::InvalidInput);
        };
        let access = operation.access();
        let params: Vec<_> = [
            access.source(),
            access.destination(),
            input[7].layout().shape()[0],
            u32::from(access.is_active()),
            query[0],
            query[1],
            state[0],
            state[1],
            state[2],
            scratch.blocks(),
            scratch.stride(),
        ]
        .into_iter()
        .chain(scratch.offsets())
        .chain([panel.unwrap_or(32)])
        .flat_map(u32::to_ne_bytes)
        .collect();
        let params = arguments.write(&params)?;
        let flag = arguments.write(&error_state())?;
        let constants = [
            (0, dtype_code(input[0].layout().dtype())),
            (1, dtype_code(input[1].layout().dtype())),
            (2, dtype_code(dispatch.output().layout().dtype())),
            (3, dtype_code(input[2].layout().dtype())),
            (4, dtype_code(input[3].layout().dtype())),
            (5, dtype_code(input[4].layout().dtype())),
        ];
        for (index, tensor) in tensors.iter().enumerate() {
            bindings.bind(table, index, &tensor.buffer);
        }
        bindings.bind(table, tensors.len(), &layouts);
        bindings.bind(table, tensors.len() + 1, &params);
        bindings.bind(table, tensors.len() + 2, &flag);
        set_argument_table(encoder, table);
        for (stage, name) in super::super::delta_block_kernels(dispatch)
            .into_iter()
            .enumerate()
        {
            #[cfg(test)]
            let name = if let Some(test) = test {
                if stage != test.stage {
                    continue;
                }
                test.kernel
            } else {
                name
            };
            if stage > 0 {
                super::super::encode_dispatch_barrier(encoder);
                super::super::write_internal_timestamp(encoder, timestamps, internal_timestamp)?;
            }
            let pipeline = self.pipelines.get(name, &constants)?;
            let (groups, threads) = stage_geometry(
                stage,
                scratch,
                state,
                dispatch.output().layout().element_count(),
                panel,
            )?;
            if pipeline.maxTotalThreadsPerThreadgroup() < threads.width {
                return Err(BackendError::UnsupportedOperation);
            }

            dispatch_threadgroups(&pipeline, encoder, groups, threads)?;
        }
        Ok((vec![layouts, params, flag.clone()], flag))
    }
}

fn stage_geometry(
    stage: usize,
    scratch: DeltaBlockScratch,
    dimensions: &[u32],
    output: u64,
    panel: Option<u32>,
) -> Result<(MTLSize, MTLSize), BackendError> {
    let size = |width, height, depth| -> Result<MTLSize, BackendError> {
        Ok(MTLSize {
            width: usize::try_from(width).map_err(|_| BackendError::InvalidInput)?,
            height: usize::try_from(height).map_err(|_| BackendError::InvalidInput)?,
            depth: usize::try_from(depth).map_err(|_| BackendError::InvalidInput)?,
        })
    };
    let [heads, keys, values]: [u32; 3] = dimensions
        .try_into()
        .map_err(|_| BackendError::InvalidInput)?;
    let mut threads = 128;
    let groups = if let Some(panel) = panel {
        let heads = u64::from(heads);
        let blocks = u64::from(scratch.blocks());
        match stage {
            0 => size(
                2,
                2,
                blocks
                    .checked_mul(heads)
                    .ok_or(BackendError::InvalidInput)?,
            )?,
            1 => size(
                (u64::from(keys) + u64::from(values)).div_ceil(32),
                heads,
                blocks,
            )?,
            2 => {
                threads = u64::from(panel / 8 * 32);
                size(u64::from(values.div_ceil(panel)), heads, 1)?
            }
            _ => size(
                u64::from(values.div_ceil(32)),
                2,
                blocks
                    .checked_mul(heads)
                    .ok_or(BackendError::InvalidInput)?,
            )?,
        }
    } else {
        size(
            stage_work(stage, scratch, dimensions, output)?.div_ceil(128),
            1,
            1,
        )?
    };
    Ok((groups, size(threads, 1, 1)?))
}

fn stage_work(
    stage: usize,
    scratch: DeltaBlockScratch,
    dimensions: &[u32],
    output: u64,
) -> Result<u64, BackendError> {
    let [heads, keys, values]: [u32; 3] = dimensions
        .try_into()
        .map_err(|_| BackendError::InvalidInput)?;
    let block = u64::try_from(DeltaBlockScratch::TOKENS).map_err(|_| BackendError::InvalidInput)?;
    match stage {
        0 => u64::from(scratch.blocks())
            .checked_mul(u64::from(heads))
            .and_then(|n| n.checked_mul(block))
            .and_then(|n| n.checked_mul(block)),
        1 => u64::from(scratch.blocks())
            .checked_mul(u64::from(heads))
            .and_then(|n| n.checked_mul(u64::from(keys).checked_add(u64::from(values))?)),
        2 => u64::from(heads).checked_mul(u64::from(values)),
        _ => Some(output),
    }
    .ok_or(BackendError::InvalidInput)
}
