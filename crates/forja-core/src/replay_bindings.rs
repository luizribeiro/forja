use crate::{BufferId, CommandList, Op, OpError, StateAccess, StateOp, Tensor};

/// Trusted per-replay replacements for protected views and authentication identities.
/// Bindings do not add parameters to a guest's graph parameter space.
#[derive(Clone, Debug, Default)]
pub struct ReplayBindings {
    views: Vec<(Tensor, Tensor)>,
    buffers: Vec<(BufferId, BufferId)>,
    states: Vec<StateBinding>,
}

#[derive(Clone, Debug)]
struct StateBinding {
    table: Tensor,
    access: StateAccess,
    views: Vec<(Tensor, Tensor)>,
}

fn compatible(old: &Tensor, new: &Tensor) -> bool {
    old.layout().dtype() == new.layout().dtype()
        && old.layout().shape() == new.layout().shape()
        && old.layout().strides() == new.layout().strides()
        && old.is_writable() == new.is_writable()
}

impl ReplayBindings {
    /// Replaces one protected view without changing its scalar geometry or access rights.
    ///
    /// # Errors
    /// Refuses incompatible geometry or ambiguous replacements.
    pub fn view(&mut self, old: Tensor, new: Tensor) -> Result<(), OpError> {
        if !compatible(&old, &new) || self.views.iter().any(|(existing, _)| existing == &old) {
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

    /// Rebinds one lane's state operands by its captured authentication table.
    ///
    /// # Errors
    /// Refuses incompatible views or ambiguous table/operand replacements.
    pub fn state_table(
        &mut self,
        table: Tensor,
        access: StateAccess,
        views: Vec<(Tensor, Tensor)>,
    ) -> Result<(), OpError> {
        if table.layout().dtype() != crate::DType::U32
            || table.layout().shape() != [2, 2]
            || self.states.iter().any(|state| state.table == table)
            || views.iter().enumerate().any(|(index, (old, new))| {
                !compatible(old, new) || views[..index].iter().any(|(existing, _)| existing == old)
            })
        {
            return Err(OpError::InvalidKernelParameter);
        }
        self.states.push(StateBinding {
            table,
            access,
            views,
        });
        Ok(())
    }

    /// Combines independent lane and bank bindings without partially changing this set on error.
    ///
    /// # Errors
    /// Refuses conflicting replacements.
    pub fn extend(&mut self, other: Self) -> Result<(), OpError> {
        let mut combined = self.clone();
        for (old, new) in other.views {
            combined.view(old, new)?;
        }
        for (old, new) in other.buffers {
            combined.buffer(old, new)?;
        }
        for state in other.states {
            combined.state_table(state.table, state.access, state.views)?;
        }
        *self = combined;
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
            let state = if let Op::State(operation) = dispatch.op() {
                let [_, table, _, _] = operation.input_suffix(dispatch.inputs())?;
                self.states.iter().find(|state| state.table == *table)
            } else {
                None
            };

            let mut dispatch = dispatch.rebase_tensors(|tensor| {
                if let Some((_, replacement)) = state
                    .and_then(|state| state.views.iter().find(|(old, _)| old == tensor))
                    .or_else(|| self.views.iter().find(|(old, _)| old == tensor))
                {
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
                && let Some(state) = state
            {
                let operation = match operation {
                    StateOp::DeltaRule(_) => StateOp::DeltaRule(state.access),
                    StateOp::DeltaRuleBlock(_) => StateOp::DeltaRuleBlock(state.access),
                    StateOp::CausalDepthwiseConv(_) => StateOp::CausalDepthwiseConv(state.access),
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
        shaped(id, DType::F32, &[size])
    }

    fn shaped(id: u64, dtype: DType, shape: &[u32]) -> Tensor {
        let bytes = shape
            .iter()
            .try_fold(dtype.byte_size(), |bytes, dimension| {
                bytes.checked_mul(u64::from(*dimension))
            })
            .unwrap();
        let buffer = BufferId::new(id, 1, bytes);
        Tensor::from_allocation(
            buffer,
            Layout::contiguous(dtype, 0, shape.to_vec(), bytes).unwrap(),
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
    #[test]
    fn lane_tables_disambiguate_identical_captured_state_views() {
        use crate::{StateLayout, StatePlane, StatePool};
        let pool = StatePool::new(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
            2,
            65536,
        )
        .unwrap();
        let mut lanes = [pool.reserve().unwrap(), pool.reserve().unwrap()];
        let access = lanes
            .iter_mut()
            .map(|lane| {
                let tx = lane.begin(1).unwrap();
                StateAccess::from_slot(&lane.bind(&tx).unwrap()).unwrap()
            })
            .collect::<Vec<_>>();
        let input = shaped(1, DType::F32, &[1, 7]);
        let weights = shaped(2, DType::F32, &[4, 7]);
        let previous = shaped(20, DType::F32, &[3, 7]);
        let next = shaped(21, DType::F32, &[3, 7]);
        let tables = [
            shaped(30, DType::U32, &[2, 2]),
            shaped(31, DType::U32, &[2, 2]),
        ];
        let generations = shaped(32, DType::U32, &[6]);
        let activity = shaped(33, DType::U32, &[1]);
        let mut commands = CommandList::new();
        for (lane, table) in tables.iter().enumerate() {
            commands
                .dispatch_many(
                    Op::State(StateOp::CausalDepthwiseConv(access[0])),
                    &[&input, &weights, &previous, table, &generations, &activity],
                    &[
                        &shaped(u64::try_from(lane).unwrap() + 3, DType::F32, &[1, 7]),
                        &next,
                    ],
                )
                .unwrap();
        }
        assert_eq!(crate::required_barriers(&commands), [false, true]);
        let views = |base| {
            vec![
                (previous.clone(), shaped(base, DType::F32, &[3, 7])),
                (next.clone(), shaped(base + 1, DType::F32, &[3, 7])),
            ]
        };
        let mut bindings = ReplayBindings::default();
        bindings
            .state_table(tables[0].clone(), access[0], views(40))
            .unwrap();
        let mut other = ReplayBindings::default();
        other
            .state_table(tables[1].clone(), access[1], views(50))
            .unwrap();
        bindings.extend(other).unwrap();
        let rebound = bindings.apply(commands).unwrap();
        for (lane, base) in [40, 50].into_iter().enumerate() {
            assert_eq!(
                rebound.dispatches()[lane].inputs()[2],
                shaped(base, DType::F32, &[3, 7])
            );
            assert_eq!(
                rebound.dispatches()[lane].outputs()[1],
                shaped(base + 1, DType::F32, &[3, 7])
            );
            assert_eq!(
                rebound.dispatches()[lane].op(),
                Op::State(StateOp::CausalDepthwiseConv(access[lane]))
            );
        }
        assert_eq!(crate::required_barriers(&rebound), [false, false]);
    }
}
