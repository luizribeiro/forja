use forja_core::{BackendError, Dispatch, StateOp, Tensor};

use super::{CpuBackend, cpu_page_table, decode, decode_u32, execution_usize};

impl CpuBackend {
    pub(super) fn execute_state(
        &self,
        dispatch: &Dispatch,
        operation: StateOp,
    ) -> Result<(), BackendError> {
        let inputs = dispatch.inputs();
        let metadata = inputs.len() - 3;
        if decode_u32(&self.read(&inputs[metadata + 2])?).ok_or(BackendError::InvalidInput)?[0] == 0
        {
            return self.write_output(
                dispatch.output(),
                &vec![0.0; execution_usize(dispatch.output().layout().element_count())?],
            );
        }
        let table = cpu_page_table(self, &inputs[metadata], &inputs[metadata + 1])?;
        let access = operation.access();
        if table
            .resolve(0)
            .map_err(|_| BackendError::ExecutionFailed)?
            != access.source()
            || table
                .resolve(1)
                .map_err(|_| BackendError::ExecutionFailed)?
                != access.destination()
        {
            return Err(BackendError::ExecutionFailed);
        }
        let values = inputs[..metadata]
            .iter()
            .map(|tensor| {
                decode(&self.read(tensor)?, tensor.layout().dtype())
                    .ok_or(BackendError::ExecutionFailed)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (output, state) = match operation {
            StateOp::DeltaRule(_) => delta_rule(&values, &inputs[5])?,
            StateOp::CausalDepthwiseConv(_) => return Err(BackendError::UnsupportedOperation),
        };
        self.write_output(&dispatch.outputs()[0], &output)?;
        self.write_output(&dispatch.outputs()[1], &state)
    }
}

fn delta_rule(
    values: &[Vec<f32>],
    state_tensor: &Tensor,
) -> Result<(Vec<f32>, Vec<f32>), BackendError> {
    let shape = state_tensor.layout().shape();
    let heads = execution_usize(shape[0])?;
    let keys = execution_usize(shape[1])?;
    let width = execution_usize(shape[2])?;
    let q_heads = values[0].len() / keys;
    let mut state = values[5].clone();
    let mut output = vec![0.0; values[2].len()];
    for head in 0..heads {
        let q_head = head / (heads / q_heads);
        for value in 0..width {
            let mut prediction = 0.0;
            for key in 0..keys {
                let index = (head * keys + key) * width + value;
                state[index] *= values[3][head];
                prediction += state[index] * values[1][q_head * keys + key];
            }
            let delta = (values[2][head * width + value] - prediction) * values[4][head];
            for key in 0..keys {
                let index = (head * keys + key) * width + value;
                state[index] += values[1][q_head * keys + key] * delta;
                output[head * width + value] += state[index] * values[0][q_head * keys + key];
            }
        }
    }
    Ok((output, state))
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{Backend, CommandList, DType, Op, Submission};

    #[test]
    fn delta_update_authenticates_before_reading_or_writing_state() {
        let backend = CpuBackend::new();
        let (access, specs) = forja_testing::state_test_metadata().unwrap();
        let mut inputs = Vec::new();
        for (shape, values) in [
            (&[1, 2][..], vec![1.0, 2.0]),
            (&[1, 2], vec![0.5, 1.0]),
            (&[1, 1], vec![3.0]),
            (&[1], vec![0.5]),
            (&[1], vec![0.25]),
            (&[1, 2, 1], vec![2.0, 4.0]),
        ] {
            let tensor = backend.alloc(DType::F32, shape).unwrap();
            backend
                .write(
                    &tensor,
                    &values
                        .into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )
                .unwrap();
            inputs.push(tensor);
        }
        for spec in specs {
            let tensor = backend
                .alloc(spec.dtype(), spec.allocation_shape())
                .unwrap();
            let bytes = forja_testing::generated_tensor_bytes(
                &spec,
                &mut forja_testing::DeterministicValues::new(1),
            )
            .unwrap();
            backend.write(&tensor, &bytes).unwrap();
            inputs.push(tensor);
        }
        let y = backend.alloc(DType::F32, &[1, 1]).unwrap();
        let state = backend.alloc(DType::F32, &[1, 2, 1]).unwrap();
        let run = || {
            let mut commands = CommandList::new();
            commands
                .dispatch_many(
                    Op::State(StateOp::DeltaRule(access)),
                    &inputs.iter().collect::<Vec<_>>(),
                    &[&y, &state],
                )
                .unwrap();
            backend.submit(commands).unwrap().wait()
        };
        run().unwrap();
        assert_eq!(
            decode(&backend.read(&y).unwrap(), DType::F32).unwrap(),
            vec![5.3125]
        );
        backend.write(&inputs[7], &[0; 12]).unwrap();
        let before = backend.read(&state).unwrap();
        assert!(run().is_err());
        assert_eq!(before, backend.read(&state).unwrap());
        backend.write(&inputs[8], &0_u32.to_le_bytes()).unwrap();
        run().unwrap();
        assert_eq!(backend.read(&y).unwrap(), [0; 4]);
        assert_eq!(before, backend.read(&state).unwrap());
    }
}
