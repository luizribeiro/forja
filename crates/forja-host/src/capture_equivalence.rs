use crate::{
    Backend, BatchPhase, BatchSelection, BatchSlotEntry, CommandListEntry, Host, compute,
    graph_error, invalid_handle, state::StateCharge,
};
use forja_core::{Dispatch, GraphTemplate, Op, OperationKind, program::ProgramHash};
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

#[derive(Clone, Debug, PartialEq)]
struct ArithmeticDispatch {
    arithmetic: Arithmetic,
}

#[derive(Debug)]
pub(crate) struct DecodeReference {
    _graph: u32,
    _summary: ArithmeticSummary,
    _charge: Arc<StateCharge>,
}

#[derive(Debug)]
pub(crate) struct ArithmeticSummary {
    dispatches: Vec<ArithmeticDispatch>,
}
impl ArithmeticSummary {
    fn byte_size(&self) -> Result<u64, compute::Error> {
        u64::try_from(self.dispatches.capacity())
            .ok()
            .and_then(|count| count.checked_mul(size_of::<ArithmeticDispatch>() as u64))
            .and_then(|bytes| bytes.checked_add(size_of::<DecodeReference>() as u64))
            .ok_or_else(|| invalid_handle("arithmetic summary size overflowed"))
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
                _graph: graph,
                _summary: arithmetic.summary,
                _charge: arithmetic.charge,
            });
        }
        Ok(())
    }

    fn arithmetic_dispatches(
        graph: &GraphTemplate,
        entry: &CommandListEntry,
    ) -> Result<Vec<ArithmeticDispatch>, compute::Error> {
        let candidates = graph
            .backend_dispatch_candidates()
            .map_err(|error| graph_error(&error))?;
        candidates
            .into_iter()
            .map(|(index, _, dispatch)| {
                let _ = (index, entry);
                Ok(ArithmeticDispatch {
                    arithmetic: Arithmetic::of(&dispatch),
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
