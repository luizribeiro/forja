use crate::BufferKind;
use crate::{
    Backend, BatchPhase, BatchSelection, BatchSlotEntry, CommandListEntry, Host, compute,
    graph_error, invalid_handle, state::StateCharge,
};
use forja_core::{BufferId, Tensor};
use forja_core::{Dispatch, GraphTemplate, Op, OperationKind, program::ProgramHash};
use std::collections::HashSet;
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
}

struct CaptureResources {
    weights: HashSet<BufferId>,
}

impl CaptureResources {
    fn key(&self, tensor: &Tensor) -> Option<ArithmeticResource> {
        let buffer = tensor.buffer();
        if self.weights.contains(&buffer) {
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
    fn capture_resources(entry: &CommandListEntry) -> CaptureResources {
        CaptureResources {
            weights: entry
                .retained
                .iter()
                .filter(|entry| !matches!(entry.buffer.kind, BufferKind::Allocated { .. }))
                .map(|entry| entry.tensor.buffer())
                .collect(),
        }
    }
    fn arithmetic_dispatches(
        graph: &GraphTemplate,
        entry: &CommandListEntry,
    ) -> Result<Vec<ArithmeticDispatch>, compute::Error> {
        let resources = Self::capture_resources(entry);
        let candidates = graph
            .backend_dispatch_candidates()
            .map_err(|error| graph_error(&error))?;
        candidates
            .into_iter()
            .map(|(index, _, dispatch)| {
                let _ = (index, entry);
                let reads = resources.reads(&dispatch);
                Ok(ArithmeticDispatch {
                    arithmetic: Arithmetic::of(&dispatch),
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
        let dispatches = Self::arithmetic_dispatches(graph, entry)?;
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
        let resources = Host::<forja_cpu::CpuBackend>::capture_resources(entry);
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
}
