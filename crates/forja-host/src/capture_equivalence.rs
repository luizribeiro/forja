use crate::BufferKind;
use crate::{
    Backend, BatchPhase, BatchSelection, BatchSlotEntry, CommandListEntry, Host, compute,
    graph_error, invalid_handle, state::StateCharge,
};
use forja_core::{BufferId, Tensor};
use forja_core::{Dispatch, GraphTemplate, Op, OperationKind, program::ProgramHash};
use std::collections::HashSet;
use std::{collections::HashMap, ops::RangeInclusive};
use std::{mem::Discriminant, sync::Arc};
use wasmtime::component::Resource;

#[derive(Clone, Debug, PartialEq)]
struct Arithmetic {
    kind: OperationKind,
    operation: Discriminant<Op>,
    class: Option<&'static str>,
    program: Option<ProgramHash>,
    epilogue: Option<ProgramHash>,
}

impl Arithmetic {
    fn of(dispatch: &Dispatch) -> Self {
        #[cfg(target_os = "macos")]
        let class = dispatch
            .backend_data::<forja_metal::ValidatedVariant>()
            .map(|variant| variant.reduction_class().unwrap_or(variant.name()));
        #[cfg(not(target_os = "macos"))]
        let class = None;
        Self {
            kind: OperationKind::of(dispatch.op()),
            operation: std::mem::discriminant(&dispatch.op()),
            class,
            program: dispatch
                .bound_program()
                .map(forja_core::program::BoundProgram::content_hash),
            epilogue: dispatch
                .prepared_epilogue()
                .map(|epilogue| epilogue.validated().content_hash()),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ArithmeticResource {
    Weight {
        buffer: BufferId,
        start: u64,
        end: u64,
    },
    Plane(BufferId),
}

struct CaptureResources {
    weights: HashSet<BufferId>,
    planes: HashSet<BufferId>,
}

impl CaptureResources {
    fn key(&self, tensor: &Tensor) -> Option<ArithmeticResource> {
        let buffer = tensor.buffer();
        if self.planes.contains(&buffer) {
            Some(ArithmeticResource::Plane(buffer))
        } else if self.weights.contains(&buffer) {
            let span = tensor.layout().byte_span();
            Some(ArithmeticResource::Weight {
                buffer,
                start: span.start,
                end: span.end,
            })
        } else {
            None
        }
    }

    fn reads(&self, dispatch: &Dispatch) -> Vec<ArithmeticResource> {
        let mut resources = Vec::new();
        // Paged append kernels read their read/write plane outputs as well as ordinary inputs.
        for tensor in dispatch.inputs().iter().chain(dispatch.outputs()) {
            if let Some(key) = self.key(tensor)
                && !resources.contains(&key)
            {
                resources.push(key);
            }
        }
        resources
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ArithmeticDispatch {
    arithmetic: Arithmetic,
    positions: RangeInclusive<u32>,
    resources: Vec<ArithmeticResource>,
}

#[derive(Debug)]
pub(crate) struct DecodeReference {
    graph: u32,
    _summary: ArithmeticSummary,
    _charge: Arc<StateCharge>,
}

#[derive(Debug)]
pub(crate) struct ArithmeticSummary {
    dispatches: Vec<ArithmeticDispatch>,
}
impl ArithmeticSummary {
    fn byte_size(&self) -> Result<u64, compute::Error> {
        let dispatches = &self.dispatches;
        let bytes = dispatches
            .iter()
            .try_fold(size_of::<DecodeReference>() as u64, |sum, dispatch| {
                u64::try_from(dispatch.resources.capacity())
                    .ok()?
                    .checked_mul(size_of::<ArithmeticResource>() as u64)?
                    .checked_add(sum)
            })
            .and_then(|bytes| {
                u64::try_from(dispatches.capacity())
                    .ok()?
                    .checked_mul(size_of::<ArithmeticDispatch>() as u64)?
                    .checked_add(bytes)
            });
        bytes.ok_or_else(|| invalid_handle("arithmetic summary size overflowed"))
    }
}

fn fixed_positions(
    space: Option<&forja_core::ParamSpace>,
    rows: u32,
) -> Result<RangeInclusive<u32>, compute::Error> {
    let Some(range) = space.and_then(|space| space.ranges().first()) else {
        // A fixed graph without a position parameter has no key boundary to extend.
        return Ok(0..=u32::MAX);
    };
    let end = rows
        .checked_sub(1)
        .and_then(|offset| range.end().checked_add(offset))
        .ok_or_else(|| invalid_handle("capture row positions overflowed"))?;
    Ok(*range.start()..=end)
}
pub(crate) struct CapturedArithmetic {
    slot: u32,
    summary: ArithmeticSummary,
    charge: Arc<StateCharge>,
}
impl<B: Backend + 'static> Host<B> {
    pub(crate) fn cache_capture_arithmetic(
        &mut self,
        arithmetic: Option<CapturedArithmetic>,
        graph: u32,
    ) -> Result<(), compute::Error> {
        if let Some(arithmetic) = arithmetic {
            self.table
                .get_mut(&Resource::<BatchSlotEntry>::new_borrow(arithmetic.slot))
                .map_err(invalid_handle)?
                .decode_arithmetic = Some(DecodeReference {
                graph,
                _summary: arithmetic.summary,
                _charge: arithmetic.charge,
            });
        }
        Ok(())
    }
    pub(crate) fn retire_capture_arithmetic(
        &mut self,
        scope: Option<&crate::slot_capture::SlotCapture>,
        graph: u32,
    ) {
        if let Some(scope) = scope
            && let Ok(slot) = self
                .table
                .get_mut(&Resource::<BatchSlotEntry>::new_borrow(scope.slot))
            && slot
                .decode_arithmetic
                .as_ref()
                .is_some_and(|reference| reference.graph == graph)
        {
            slot.decode_arithmetic = None;
        }
    }
    fn capture_resources(
        &self,
        slot: u32,
        entry: &CommandListEntry,
    ) -> Result<CaptureResources, compute::Error> {
        let mut planes = self
            .state_slots
            .get(&slot)
            .ok_or_else(|| invalid_handle("capture state slot is missing"))?
            .plane_buffers()
            .collect::<HashSet<_>>();
        for handle in &self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(slot))
            .map_err(invalid_handle)?
            .planes
        {
            planes.insert(self.entry(&Resource::new_borrow(*handle))?.tensor.buffer());
        }
        Ok(CaptureResources {
            weights: entry
                .retained
                .iter()
                .filter(|entry| !matches!(entry.buffer.kind, BufferKind::Allocated { .. }))
                .map(|entry| entry.tensor.buffer())
                .collect(),
            planes,
        })
    }
    fn arithmetic_dispatches(
        &self,
        slot: u32,
        graph: &GraphTemplate,
        entry: &CommandListEntry,
    ) -> Result<Vec<ArithmeticDispatch>, compute::Error> {
        let resources = self.capture_resources(slot, entry)?;
        let rows = entry
            .slot_capture
            .as_ref()
            .filter(|scope| scope.shape.phase == BatchPhase::Verify)
            .map_or(1, |scope| scope.shape.tokens);
        let fixed = fixed_positions(entry.space.as_ref(), rows)?;
        let baseline = entry
            .slot_capture
            .as_ref()
            .is_some_and(|scope| scope.shape.phase == BatchPhase::Decode);
        let mut ordinals = HashMap::<usize, usize>::new();
        let candidates = graph
            .backend_dispatch_candidates()
            .map_err(|error| graph_error(&error))?;
        if baseline {
            self.validate_decode_output_roles(slot, &candidates)?;
        }
        candidates
            .into_iter()
            .map(|(index, _, dispatch)| {
                if let Some(key) = graph.dispatch_rule_key(index)
                    && (key.parameter != 0 || (baseline && key.row_offset != 0))
                {
                    return Err(invalid_handle(
                        "decode reference rules require the unshifted position parameter",
                    ));
                }
                let ordinal = ordinals.entry(index).or_default();
                let positions = graph
                    .dispatch_rule_intervals(index)
                    .map_err(|error| graph_error(&error))?
                    .map_or_else(
                        || Ok(fixed.clone()),
                        |intervals| {
                            intervals
                                .get(*ordinal)
                                .cloned()
                                .ok_or_else(|| invalid_handle("capture rule interval is missing"))
                        },
                    )?;
                *ordinal += 1;
                let reads = resources.reads(&dispatch);
                Ok(ArithmeticDispatch {
                    arithmetic: Arithmetic::of(&dispatch),
                    positions,
                    resources: reads,
                })
            })
            .collect()
    }
    pub(crate) fn capture_arithmetic(
        &self,
        graph: &GraphTemplate,
        entry: &CommandListEntry,
    ) -> Result<Option<CapturedArithmetic>, compute::Error> {
        let Some(scope) = &entry.slot_capture else {
            return Ok(None);
        };
        let verify = scope.shape.phase == BatchPhase::Verify;
        let baseline = scope.shape.phase == BatchPhase::Decode
            && scope.shape.bucket == 1
            && scope.shape.selection == BatchSelection::Combined;
        if !verify && !baseline {
            return Ok(None);
        }
        scope.validate(self, false)?;
        let slot = self
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(scope.slot))
            .map_err(invalid_handle)?;
        if verify && slot.decode_arithmetic.is_none() {
            return Err(invalid_handle(
                "verify capture requires prior bucket-one combined decode",
            ));
        }
        if slot.decode_arithmetic.is_some() {
            return Ok(None);
        }
        let dispatches = self.arithmetic_dispatches(scope.slot, graph, entry)?;
        let summary = ArithmeticSummary { dispatches };
        let bytes = summary.byte_size()?;
        let charge = StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.live_bytes),
            self.limits.live_bytes,
            bytes,
        )?;
        Ok(Some(CapturedArithmetic {
            slot: scope.slot,
            summary,
            charge: Arc::new(charge),
        }))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn weight_resource_metadata_distinguishes_regions_and_preserves_aliases() {
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            crate::Limits::new(65536, 64, 4096, 16, 4096),
        );
        let (path, weights) = crate::tests::open_test_capture_weights(&mut host);
        let commands = host.command_list().unwrap();
        crate::tests::record_test_weight_operations(
            &mut host,
            &commands,
            &[weights[0].rep(), weights[1].rep()],
            false,
        );
        let entry = host.table.get(&commands).unwrap();
        let resources = CaptureResources {
            weights: entry
                .retained
                .iter()
                .filter(|entry| !matches!(entry.buffer.kind, BufferKind::Allocated { .. }))
                .map(|entry| entry.tensor.buffer())
                .collect(),
            planes: HashSet::new(),
        };
        let alias = host
            .view(&weights[0], crate::compute::ViewOp::Permute(vec![0]))
            .unwrap();
        assert_eq!(
            resources.key(&host.entry(&weights[0]).unwrap().tensor),
            resources.key(&host.entry(&alias).unwrap().tensor)
        );
        assert_ne!(
            resources.key(&host.entry(&weights[0]).unwrap().tensor),
            resources.key(&host.entry(&weights[1]).unwrap().tensor)
        );
        std::fs::remove_file(path).unwrap();
    }
    #[test]
    fn capture_binding_recognizes_actual_state_and_paged_planes() {
        let (mut host, _pool, slot) =
            crate::tests::verify_input_fixture(forja_cpu::CpuBackend::new(), 3, 1);
        let commands = host.command_list().unwrap();
        let state = host.state_slots[&slot.rep()]
            .plane_buffers()
            .next()
            .unwrap();
        let page = host
            .entry(&Resource::new_borrow(
                host.table.get(&slot).unwrap().planes[0],
            ))
            .unwrap()
            .tensor
            .buffer();
        let entry = host.table.get(&commands).unwrap();
        let resources = host.capture_resources(slot.rep(), entry).unwrap();
        let view = |buffer, offset| {
            host.backend
                .tensor(
                    buffer,
                    forja_core::Layout::new(
                        forja_core::DType::F32,
                        offset,
                        vec![7],
                        vec![1],
                        buffer.byte_len(),
                    )
                    .unwrap(),
                )
                .unwrap()
        };
        assert_eq!(
            resources.key(&view(state, 0)).unwrap(),
            resources.key(&view(state, 7)).unwrap()
        );
        assert_ne!(
            resources.key(&view(state, 0)).unwrap(),
            resources.key(&view(page, 0)).unwrap()
        );
    }
    #[test]
    fn fixed_verify_dispatch_covers_the_next_rule_key() {
        let space = forja_core::ParamSpace::new(std::iter::once(509..=509).collect()).unwrap();
        assert_eq!(fixed_positions(Some(&space), 1).unwrap(), 509..=509);
        assert_eq!(fixed_positions(Some(&space), 2).unwrap(), 509..=510);
        let end =
            forja_core::ParamSpace::new(std::iter::once(u32::MAX..=u32::MAX).collect()).unwrap();
        assert!(fixed_positions(Some(&end), 2).is_err());
        assert_eq!(fixed_positions(None, 3).unwrap(), 0..=u32::MAX);
    }
}
