mod rows;

use crate::{
    Backend, BatchPhase, BatchSelection, BatchShape, BatchSlotEntry, CommandListEntry, Host,
    PagedBinding, SharedStateTrace, compute, invalid_handle, state::StateCharge,
};
use std::sync::{Arc, Weak, atomic::Ordering};
use wasmtime::component::Resource;

#[derive(Clone, Debug)]
pub(crate) struct SlotCapture {
    pub(crate) slot: u32,
    pub(crate) shape: BatchShape,
    binding: Weak<PagedBinding>,
    pub(crate) trace: SharedStateTrace,
    _charge: Arc<StateCharge>,
}

#[derive(Clone, Copy)]
pub(crate) struct PromptReplayLane {
    slot: u32,
    lane: usize,
}
impl PromptReplayLane {
    pub(crate) fn map(remap: Option<Self>, slot: u32, lane: usize) -> usize {
        remap
            .filter(|remap| remap.slot == slot && lane == 0)
            .map_or(lane, |remap| remap.lane)
    }
}

pub(crate) fn capture_shape(
    kind: crate::paged_storage::CaptureKind,
    selection: crate::paged_storage::CaptureSelection,
) -> BatchShape {
    use crate::paged_storage::{CaptureKind, CaptureSelection};
    let (phase, bucket, tokens) = match kind {
        CaptureKind::Prompt => (BatchPhase::Prompt, 1, 1),
        CaptureKind::Decode(bucket) => (BatchPhase::Decode, bucket, 1),
        CaptureKind::Verify(rows) => (BatchPhase::Verify, 1, rows),
    };
    BatchShape {
        phase,
        bucket,
        tokens,
        selection: match selection {
            CaptureSelection::Combined => BatchSelection::Combined,
            CaptureSelection::Forward => BatchSelection::Forward,
            CaptureSelection::Masked => BatchSelection::Masked,
        },
    }
}

impl SlotCapture {
    pub(crate) fn prefill(&self) -> bool {
        self.shape.phase == BatchPhase::Prompt
    }

    pub(crate) fn prompt_lane<B: Backend + 'static>(
        &self,
        host: &Host<B>,
        numerical: bool,
    ) -> Result<Option<PromptReplayLane>, compute::Error> {
        self.validate(host, numerical)?;
        if !self.prefill() {
            return Ok(None);
        }
        let slot = host
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(self.slot))
            .map_err(invalid_handle)?;
        let mut active = slot.binding.activity[slot.active_bank]
            .iter()
            .enumerate()
            .filter(|(_, active)| active.load(Ordering::Acquire));
        let lane = active.next().map(|(lane, _)| lane);
        if active.next().is_some() {
            return Err(invalid_handle(
                "prompt replay requires at most one active lane",
            ));
        }
        Ok(lane.map(|lane| PromptReplayLane {
            slot: self.slot,
            lane,
        }))
    }

    pub(crate) fn validate<B: Backend + 'static>(
        &self,
        host: &Host<B>,
        numerical: bool,
    ) -> Result<(), compute::Error> {
        let slot = host
            .table
            .get(&Resource::<BatchSlotEntry>::new_borrow(self.slot))
            .map_err(invalid_handle)?;
        let inputs = slot
            .inputs
            .as_ref()
            .ok_or_else(|| invalid_handle("capture slot has no state inputs"))?;
        if !self.binding.ptr_eq(&Arc::downgrade(&slot.binding))
            || (numerical
                && inputs.banks[slot.active_bank]
                    .prompt_phase
                    .load(Ordering::Acquire)
                    != self.prefill())
        {
            return Err(invalid_handle(
                "capture slot identity or numerical phase differs",
            ));
        }
        Ok(())
    }
}

impl<B: Backend + Send + Sync + 'static> Host<B> {
    pub(crate) fn validate_capture_shape(&self, shape: BatchShape) -> Result<(), compute::Error> {
        let declared = self.capture_shapes.iter().any(|declared| {
            if shape.phase == BatchPhase::Prompt {
                declared.phase == BatchPhase::Prompt && declared.selection == shape.selection
            } else {
                *declared == shape
            }
        });
        if !declared {
            return Err(invalid_handle(
                "capture kind and shape are not declared for serving",
            ));
        }
        Ok(())
    }

    /// Declares a graph-only slot recording with independent inactive state bindings.
    ///
    /// # Errors
    /// Refuses nonempty lists, duplicate declarations, missing state slots or exhausted quota.
    pub fn declare_slot_capture(
        &mut self,
        commands: &Resource<CommandListEntry>,
        slot: &Resource<BatchSlotEntry>,
        shape: BatchShape,
    ) -> Result<(), compute::Error> {
        let entry = self.table.get(commands).map_err(invalid_handle)?;
        if !entry.recorded.is_empty() || entry.slot_capture.is_some() {
            return Err(invalid_handle(
                "slot capture requires an empty undeclared command list",
            ));
        }
        let binding = Arc::downgrade(&self.table.get(slot).map_err(invalid_handle)?.binding);
        let charge = self.reserve_reference()?;
        let slot_entry = self.table.get(slot).map_err(invalid_handle)?;
        let handles = slot_entry.metadata[0].into_iter().chain(
            slot_entry
                .inputs
                .iter()
                .flat_map(|inputs| inputs.banks[0].handles()),
        );
        let buffers = handles
            .map(|handle| {
                self.entry(&Resource::new_borrow(handle))
                    .map(|entry| entry.tensor.buffer())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let trace = self
            .state_slots
            .get(&slot.rep())
            .ok_or_else(|| invalid_handle("capture slot has no state"))?
            .trace()?;
        let entry = self.table.get_mut(commands).map_err(invalid_handle)?;
        entry.protected.extend(buffers);
        entry.slot_capture = Some(SlotCapture {
            slot: slot.rep(),
            shape,
            binding,
            trace: trace.share(),
            _charge: charge,
        });
        Ok(())
    }
}

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn slot_capture_bank(
        &self,
        commands: &Resource<CommandListEntry>,
        slot: &Resource<BatchSlotEntry>,
    ) -> Result<Option<usize>, compute::Error> {
        match self
            .table
            .get(commands)
            .map_err(invalid_handle)?
            .slot_capture
            .as_ref()
        {
            None => Ok(None),
            Some(capture) if capture.slot == slot.rep() => Ok(Some(0)),
            Some(_) => Err(invalid_handle("paged work differs from its capture slot")),
        }
    }

    pub(crate) fn with_state_commands<T>(
        &mut self,
        usage: crate::hybrid::StateUse,
        capture: Option<&SlotCapture>,
        action: impl FnOnce(&mut crate::StateCommands<B>) -> Result<T, compute::Error>,
    ) -> Result<T, compute::Error> {
        if let Some(capture) = capture {
            if capture.slot != usage.slot
                || usage.active
                || usage.bank != 0
                || (capture.prefill() && usage.lane != 0)
                || match capture.shape.phase {
                    BatchPhase::Prompt => false,
                    BatchPhase::Decode => usage.tokens != 1,
                    BatchPhase::Verify => usage.tokens != u64::from(capture.shape.tokens),
                }
            {
                return Err(invalid_handle(
                    "state work differs from its capture declaration",
                ));
            }
            let mut trace = capture.trace.lock::<B>()?;
            return action(
                trace
                    .as_mut()
                    .ok_or_else(|| invalid_handle("state trace has retired"))?
                    .lane(usage.lane)?,
            );
        }
        let state = self
            .state_slots
            .get_mut(&usage.slot)
            .ok_or_else(|| invalid_handle("state slot is missing"))?;
        action(state.commands(usage.bank, usage.lane, usage.active)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_storage::{CaptureKind, CaptureSelection};

    #[test]
    fn captured_state_spans_require_the_declared_phase_extent() {
        use crate::hybrid::StateUse;
        let (mut host, _pool, slot) =
            crate::tests::verify_input_fixture(forja_cpu::CpuBackend::new(), 3, 8);
        for (kind, extent) in [
            (CaptureKind::Decode(1), 1),
            (CaptureKind::Verify(2), 2),
            (CaptureKind::Verify(3), 3),
        ] {
            let commands = host.command_list().unwrap();
            host.declare_slot_capture(
                &commands,
                &slot,
                capture_shape(kind, CaptureSelection::Combined),
            )
            .unwrap();
            let capture = host
                .table
                .get(&commands)
                .unwrap()
                .slot_capture
                .clone()
                .unwrap();
            let usage = StateUse {
                slot: slot.rep(),
                bank: 0,
                lane: 0,
                active: false,
                tokens: extent,
            };
            host.with_state_commands(usage, Some(&capture), |_| Ok(()))
                .unwrap();
            for tokens in [0, 1, 2, 3, 4]
                .into_iter()
                .filter(|tokens| *tokens != extent)
            {
                assert!(
                    host.with_state_commands(StateUse { tokens, ..usage }, Some(&capture), |_| Ok(
                        ()
                    ))
                    .is_err()
                );
            }
            for bad in [
                StateUse { bank: 1, ..usage },
                StateUse {
                    active: true,
                    ..usage
                },
                StateUse {
                    slot: u32::MAX,
                    ..usage
                },
            ] {
                assert!(
                    host.with_state_commands(bad, Some(&capture), |_| Ok(()))
                        .is_err()
                );
            }
            host.drop_command_list(commands).unwrap();
        }
    }

    #[test]
    fn capture_shapes_require_declared_phase_extent_and_selection() {
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            crate::Limits::new(65536, 8, 4096, 16, 4096),
        );
        let shape = capture_shape(CaptureKind::Verify(3), CaptureSelection::Combined);
        assert!(host.validate_capture_shape(shape).is_err());
        host.capture_shapes.push(shape);
        host.validate_capture_shape(shape).unwrap();
        for kind in [
            CaptureKind::Prompt,
            CaptureKind::Decode(1),
            CaptureKind::Verify(2),
            CaptureKind::Verify(9),
        ] {
            assert!(
                host.validate_capture_shape(capture_shape(kind, CaptureSelection::Combined))
                    .is_err()
            );
        }
        for selection in [CaptureSelection::Forward, CaptureSelection::Masked] {
            assert!(
                host.validate_capture_shape(capture_shape(CaptureKind::Verify(3), selection))
                    .is_err()
            );
        }
        host.capture_shapes.push(BatchShape {
            phase: BatchPhase::Prompt,
            tokens: 33,
            ..shape
        });
        host.validate_capture_shape(capture_shape(
            CaptureKind::Prompt,
            CaptureSelection::Combined,
        ))
        .unwrap();
        host.capture_shapes.push(capture_shape(
            CaptureKind::Decode(4),
            CaptureSelection::Forward,
        ));
        host.validate_capture_shape(capture_shape(
            CaptureKind::Decode(4),
            CaptureSelection::Forward,
        ))
        .unwrap();
        assert!(
            host.validate_capture_shape(capture_shape(
                CaptureKind::Decode(1),
                CaptureSelection::Forward
            ))
            .is_err()
        );
    }

    #[test]
    fn capture_kinds_preserve_geometry_and_selection() {
        for (kind, phase, bucket, tokens) in [
            (CaptureKind::Prompt, BatchPhase::Prompt, 1, 1),
            (CaptureKind::Decode(4), BatchPhase::Decode, 4, 1),
            (CaptureKind::Verify(3), BatchPhase::Verify, 1, 3),
        ] {
            for (selection, expected) in [
                (CaptureSelection::Combined, BatchSelection::Combined),
                (CaptureSelection::Forward, BatchSelection::Forward),
                (CaptureSelection::Masked, BatchSelection::Masked),
            ] {
                assert_eq!(
                    capture_shape(kind, selection),
                    BatchShape {
                        phase,
                        bucket,
                        tokens,
                        selection: expected,
                    }
                );
            }
        }
    }
}
