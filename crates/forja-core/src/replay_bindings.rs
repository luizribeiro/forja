use crate::{BufferId, CommandList, Op, OpError, StateAccess, StateOp, Tensor};

/// Trusted per-replay replacements for protected views and authentication identities.
/// Bindings do not add parameters to a guest's graph parameter space.
#[derive(Clone, Debug, Default)]
pub struct ReplayBindings {
    views: Vec<(Tensor, Tensor)>,
    buffers: Vec<(BufferId, BufferId)>,
    states: Vec<(StateAccess, StateAccess)>,
}

impl ReplayBindings {
    /// Replaces one protected view without changing its scalar geometry or access rights.
    ///
    /// # Errors
    /// Refuses incompatible geometry or ambiguous replacements.
    pub fn view(&mut self, old: Tensor, new: Tensor) -> Result<(), OpError> {
        if old.layout().dtype() != new.layout().dtype()
            || old.layout().shape() != new.layout().shape()
            || old.layout().strides() != new.layout().strides()
            || old.is_writable() != new.is_writable()
            || self.views.iter().any(|(existing, _)| existing == &old)
        {
            return Err(OpError::InvalidKernelParameter);
        }
        self.views.push((old, new));
        Ok(())
    }

    /// Rebinds every view of a fixed-size host metadata allocation to another slot bank.
    ///
    /// # Errors
    /// Refuses different allocation extents or duplicate source bindings.
    pub fn buffer(&mut self, old: BufferId, new: BufferId) -> Result<(), OpError> {
        if old.byte_len() != new.byte_len()
            || self.buffers.iter().any(|(existing, _)| *existing == old)
        {
            return Err(OpError::InvalidKernelParameter);
        }
        self.buffers.push((old, new));
        Ok(())
    }

    /// Replaces the expected authenticated input/output identities for one lane.
    ///
    /// # Errors
    /// Refuses ambiguous source bindings.
    pub fn state(&mut self, old: StateAccess, new: StateAccess) -> Result<(), OpError> {
        if self.states.iter().any(|(existing, _)| *existing == old) {
            return Err(OpError::InvalidKernelParameter);
        }
        self.states.push((old, new));
        Ok(())
    }

    /// Reports whether a prepared backend plan can be reused unchanged.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.views.is_empty() && self.buffers.is_empty() && self.states.is_empty()
    }

    /// Revalidates changed operands and rebuilds hazards after physical rebinding.
    ///
    /// # Errors
    /// Refuses invalid layouts, aliases or authentication geometry before submission.
    pub fn apply(&self, commands: CommandList) -> Result<CommandList, OpError> {
        let mut bound = CommandList::new();
        for dispatch in commands.into_dispatches() {
            let mut dispatch = dispatch.rebase_tensors(|tensor| {
                if let Some((_, replacement)) = self.views.iter().find(|(old, _)| old == tensor) {
                    return Ok(replacement.clone());
                }
                self.buffers
                    .iter()
                    .find(|(old, _)| *old == tensor.buffer())
                    .map_or_else(
                        || Ok(tensor.clone()),
                        |(_, replacement)| tensor.rebase(*replacement, 0),
                    )
            })?;
            if let Op::State(operation) = dispatch.op()
                && let Some((_, replacement)) = self
                    .states
                    .iter()
                    .find(|(old, _)| *old == operation.access())
            {
                let operation = match operation {
                    StateOp::DeltaRule(_) => StateOp::DeltaRule(*replacement),
                    StateOp::DeltaRuleBlock(_) => StateOp::DeltaRuleBlock(*replacement),
                    StateOp::CausalDepthwiseConv(_) => StateOp::CausalDepthwiseConv(*replacement),
                };
                dispatch.bind_state(operation)?;
            }
            bound.record_validated(dispatch);
        }
        Ok(bound)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Layout};

    fn tensor(id: u64, size: u32) -> Tensor {
        let buffer = BufferId::new(id, 1, u64::from(size) * 4);
        Tensor::from_allocation(
            buffer,
            Layout::contiguous(DType::F32, 0, vec![size], buffer.byte_len()).unwrap(),
            true,
        )
        .unwrap()
    }

    #[test]
    fn rebinding_revalidates_aliases_and_recomputes_cross_bank_hazards() {
        let a = tensor(1, 33);
        let b = tensor(2, 33);
        let c = tensor(3, 33);
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&a], &b).unwrap();
        commands.dispatch(Op::Copy, &[&c], &a).unwrap();
        let mut bindings = ReplayBindings::default();
        bindings.buffer(c.buffer(), b.buffer()).unwrap();
        let rebound = bindings.apply(commands).unwrap();
        assert_eq!(rebound.dispatches()[1].inputs()[0], b);
        assert_eq!(crate::required_barriers(&rebound), [false, true]);
        let mut invalid = ReplayBindings::default();
        invalid.view(a.clone(), b.clone()).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&a], &b).unwrap();
        assert!(matches!(
            invalid.apply(commands),
            Err(OpError::Aliasing { .. })
        ));
        assert!(ReplayBindings::default().view(a, tensor(4, 7)).is_err());
    }
}
