use super::{Dispatch, Tensor};

pub(super) fn chain_member(prior: &[Dispatch], candidate: &Dispatch) -> bool {
    let Some(previous) = prior.last() else {
        return false;
    };
    let Some(source) = candidate.inputs().len().checked_sub(4) else {
        return false;
    };
    let (Some(input), Some(state)) = (candidate.inputs().get(source), previous.outputs().get(1))
    else {
        return false;
    };
    if input != state {
        return false;
    }
    let overlap = |a: &Tensor, b: &Tensor| {
        a.buffer() == b.buffer() && forja_core::byte_ranges_overlap(a.layout(), b.layout())
    };
    prior.iter().enumerate().all(|(index, earlier)| {
        earlier
            .outputs()
            .iter()
            .enumerate()
            .all(|(output, tensor)| {
                candidate.inputs().iter().enumerate().all(|(input, other)| {
                    !overlap(tensor, other)
                        || (index + 1 == prior.len() && output == 1 && input == source)
                }) && candidate
                    .outputs()
                    .iter()
                    .enumerate()
                    .all(|(other_output, other)| {
                        // Matching geometry keeps a reused result on the same SIMD owner.
                        !overlap(tensor, other)
                            || (output == 0 && other_output == 0 && tensor == other)
                    })
            })
            && candidate
                .outputs()
                .iter()
                .all(|output| earlier.inputs().iter().all(|input| !overlap(output, input)))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{CommandList, DType, Op, Slice, StateOp, ViewOp};

    fn record(op: Op, inputs: &[Tensor], outputs: &[Tensor]) -> Dispatch {
        let mut commands = CommandList::new();
        commands
            .dispatch_many(
                op,
                &inputs.iter().collect::<Vec<_>>(),
                &outputs.iter().collect::<Vec<_>>(),
            )
            .unwrap();
        commands.into_dispatches().remove(0)
    }

    fn delta(backend: &forja_cpu::CpuBackend, previous: Option<&Dispatch>) -> Dispatch {
        let access = forja_testing::state_test_metadata().unwrap().0;
        let mut inputs = [
            (DType::F32, vec![1, 7]),
            (DType::F32, vec![1, 7]),
            (DType::F32, vec![1, 7]),
            (DType::F32, vec![1]),
            (DType::F32, vec![1]),
            (DType::F32, vec![1, 7, 7]),
            (DType::U32, vec![2, 2]),
            (DType::U32, vec![3]),
            (DType::U32, vec![1]),
        ]
        .map(|(dtype, shape)| backend.alloc(dtype, &shape).unwrap());
        if let Some(previous) = previous {
            inputs[5] = previous.outputs()[1].clone();
        }
        let outputs = [
            backend.alloc(DType::F32, &[1, 7]).unwrap(),
            backend.alloc(DType::F32, &[1, 7, 7]).unwrap(),
        ];
        record(Op::State(StateOp::DeltaRule(access)), &inputs, &outputs)
    }

    #[test]
    fn chained_state_edges_refuse_unrelated_tensor_dependencies() {
        let backend = forja_cpu::CpuBackend::new();
        let first = delta(&backend, None);
        let next = delta(&backend, Some(&first));
        assert_eq!(next.inputs().len(), 9);
        let prior = std::slice::from_ref(&first);
        assert!(chain_member(prior, &next));
        assert!(!chain_member(prior, &delta(&backend, None)));
        let mut inputs = next.inputs().to_vec();
        inputs[0] = first.outputs()[0].clone();
        assert!(!chain_member(
            prior,
            &record(next.op(), &inputs, next.outputs())
        ));
        inputs[0] = next.inputs()[0].clone();
        let mut outputs = next.outputs().to_vec();
        outputs[1] = first.inputs()[5].clone();
        assert!(!chain_member(prior, &record(next.op(), &inputs, &outputs)));
    }

    #[test]
    fn older_state_reads_are_refused_even_with_the_latest_carry() {
        let backend = forja_cpu::CpuBackend::new();
        let first = delta(&backend, None);
        let second = delta(&backend, Some(&first));
        let third = delta(&backend, Some(&second));
        let prior = [first, second];
        assert!(chain_member(&prior, &third));
        let mut inputs = third.inputs().to_vec();
        inputs[5] = prior[0].outputs()[1].clone();
        assert!(!chain_member(
            &prior,
            &record(third.op(), &inputs, third.outputs())
        ));
        inputs[5] = third.inputs()[5].clone();
        let row = backend
            .view(
                &prior[0].outputs()[1],
                ViewOp::Slice(vec![
                    Slice::new(0, 1, 1).unwrap(),
                    Slice::new(0, 1, 1).unwrap(),
                    Slice::new(0, 7, 1).unwrap(),
                ]),
            )
            .unwrap();
        inputs[0] = backend.view(&row, ViewOp::Reshape(vec![1, 7])).unwrap();
        assert!(!chain_member(
            &prior,
            &record(third.op(), &inputs, third.outputs())
        ));
    }

    #[test]
    fn steps_may_share_the_same_primary_output() {
        let backend = forja_cpu::CpuBackend::new();
        let first = delta(&backend, None);
        let next = delta(&backend, Some(&first));
        let mut outputs = next.outputs().to_vec();
        outputs[0] = first.outputs()[0].clone();
        let shared = record(next.op(), next.inputs(), &outputs);
        assert!(chain_member(std::slice::from_ref(&first), &shared));
    }

    #[test]
    fn incomplete_dispatches_refuse_chain_membership() {
        let backend = forja_cpu::CpuBackend::new();
        let complete = delta(&backend, None);
        let input = backend.alloc(DType::F32, &[7]).unwrap();
        let output = backend.alloc(DType::F32, &[7]).unwrap();
        let short = record(Op::Copy, &[input], &[output]);
        assert!(!chain_member(&[], &complete));
        assert!(!chain_member(std::slice::from_ref(&complete), &short));
        assert!(!chain_member(&[short], &complete));
    }
}
