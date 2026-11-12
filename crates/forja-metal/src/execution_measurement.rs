use crate::MetalBackend;
use forja_core::{BackendError, Dispatch, KernelCost, Op};
use std::{
    sync::{Arc, OnceLock},
    time::Duration,
};

/// Conservative work observed without per-dispatch profiling.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ExecutionWork {
    /// FLOPs in unconditional dense matrix products; excludes routed or masked work.
    pub matrix_flops: u64,
    /// Largest contiguous input of a dense matrix product, copy or argmax.
    /// This is a working-set footprint, not a DRAM transaction count.
    pub largest_input_bytes: u64,
}

#[cfg(test)]
thread_local! { static CALCULATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

impl ExecutionWork {
    pub(crate) fn of(dispatches: &[Dispatch]) -> Self {
        dispatches
            .iter()
            .filter_map(Self::dispatch)
            .fold(Self::default(), Self::including)
    }

    pub(crate) fn including(self, contribution: Self) -> Self {
        let Some(matrix_flops) = self.matrix_flops.checked_add(contribution.matrix_flops) else {
            return self;
        };
        Self {
            matrix_flops,
            largest_input_bytes: self
                .largest_input_bytes
                .max(contribution.largest_input_bytes),
        }
    }

    fn dispatch(dispatch: &Dispatch) -> Option<Self> {
        #[cfg(test)]
        CALCULATIONS.update(|count| count + 1);
        let matrix = matches!(dispatch.op(), Op::Matmul | Op::QuantMatmul { .. });
        if !matrix && !matches!(dispatch.op(), Op::Copy | Op::Argmax { .. }) {
            return None;
        }
        let mut work = Self {
            matrix_flops: if matrix {
                KernelCost::of(dispatch)?.flops
            } else {
                0
            },
            largest_input_bytes: 0,
        };
        for tensor in dispatch.primitive_inputs() {
            let layout = tensor.layout();
            if layout.is_contiguous() {
                work.largest_input_bytes = work.largest_input_bytes.max(
                    layout
                        .element_count()
                        .checked_mul(layout.dtype().byte_size())?,
                );
            }
        }
        Some(work)
    }
}

/// Validates completed command-buffer work and its optional unprofiled GPU duration.
pub type ExecutionValidator =
    dyn Fn(ExecutionWork, Option<Duration>) -> Result<(), BackendError> + Send + Sync;

pub(crate) struct Observation {
    pub(crate) work: ExecutionWork,
    pub(crate) validator: Arc<ExecutionValidator>,
    pub(crate) result: OnceLock<Result<(), BackendError>>,
}

impl Observation {
    pub(crate) fn check(&self, elapsed: Option<Duration>) -> Result<(), BackendError> {
        *self
            .result
            .get_or_init(|| (self.validator)(self.work, elapsed))
    }
}

impl MetalBackend {
    /// Checks every successfully completed command buffer during a host wait.
    /// Graph replay and split buffers retain their own executed work and GPU interval.
    /// The validator runs once per buffer, outside Metal callbacks; missing GPU timestamps
    /// are reported as `None` so observers can record a skipped measurement.
    #[must_use]
    pub fn with_execution_validator(mut self, validator: Arc<ExecutionValidator>) -> Self {
        self.execution_validator = Some(validator);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{
        AllocationRegistry, Backend, CommandList, DType, GraphLimits, GraphTemplate, Layout,
        ParamSpace, Submission,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn dense_work_counts_products_and_retains_a_single_input_footprint() {
        let mut registry = AllocationRegistry::new();
        let mut tensor = |shape: Vec<u32>| {
            let bytes = shape.iter().map(|&n| u64::from(n)).product::<u64>() * 4;
            let buffer = registry.insert((), bytes).unwrap();
            registry
                .tensor(
                    buffer,
                    Layout::contiguous(DType::F32, 0, shape, bytes).unwrap(),
                )
                .unwrap()
        };
        for rows in [1, 7, 33] {
            let a = tensor(vec![rows, 1024]);
            let b = tensor(vec![1024, 4097]);
            let c = tensor(vec![rows, 4097]);
            let mut commands = CommandList::new();
            for _ in 0..2 {
                commands.dispatch(Op::Matmul, &[&a, &b], &c).unwrap();
            }
            let work = ExecutionWork::of(commands.dispatches());
            assert_eq!(work.matrix_flops, 4 * u64::from(rows) * 1024 * 4097);
            assert_eq!(work.largest_input_bytes, 1024 * 4097 * 4);
        }
    }

    #[test]
    fn noncontiguous_inputs_do_not_claim_a_streaming_footprint() {
        let mut registry = AllocationRegistry::new();
        let input = registry.insert((), 7 * 33 * 4).unwrap();
        let output = registry.insert((), 7 * 33 * 4).unwrap();
        let layout = Layout::contiguous(DType::F32, 0, vec![7, 33], 7 * 33 * 4).unwrap();
        let input = registry
            .tensor(input, layout.permute(&[1, 0]).unwrap())
            .unwrap();
        let output = registry
            .tensor(output, layout.reshape(&[33, 7]).unwrap())
            .unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();
        assert_eq!(
            ExecutionWork::of(commands.dispatches()),
            ExecutionWork::default()
        );
    }

    #[test]
    fn metal_execution_validation_refuses_replay_without_profiling() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let backend = MetalBackend::new()
            .unwrap()
            .with_execution_validator(Arc::new(move |work, elapsed| {
                assert_eq!(work.largest_input_bytes, 4097 * 4);
                assert!(elapsed.is_some_and(|elapsed| !elapsed.is_zero()));
                observed.fetch_add(1, Ordering::Relaxed);
                Err(BackendError::ImpossibleRate {
                    resource: "bytes",
                    achieved: 111,
                    limit: 110,
                })
            }));
        let input = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        let output = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        let mut template =
            GraphTemplate::new(ParamSpace::new(vec![]).unwrap(), GraphLimits::default());
        template
            .dispatch(Op::Copy, &[&input.into()], &output.into())
            .unwrap();
        let graph = backend.prepare_graph(template).unwrap();
        let submission = backend.replay(&graph, vec![]).unwrap();
        for _ in 0..2 {
            assert!(matches!(
                submission.wait(),
                Err(BackendError::ImpossibleRate { .. })
            ));
        }
        assert!(submission.profile().is_none());
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn unknown_matrix_cost_is_omitted() {
        let mut registry = AllocationRegistry::new();
        let mut tensor = |shape: Vec<u32>| {
            let bytes = shape.iter().map(|&n| u64::from(n)).product::<u64>() * 4;
            let buffer = registry.insert((), bytes).unwrap();
            registry
                .tensor(
                    buffer,
                    Layout::contiguous(DType::F32, 0, shape, bytes).unwrap(),
                )
                .unwrap()
        };
        let a = tensor(vec![4097, 1 << 30]);
        let b = tensor(vec![1 << 30, 1 << 30]);
        let c = tensor(vec![4097, 1 << 30]);
        let mut commands = CommandList::new();
        commands.dispatch(Op::Matmul, &[&a, &b], &c).unwrap();
        assert!(KernelCost::of(commands.last_dispatch().unwrap()).is_none());
        assert_eq!(
            ExecutionWork::of(commands.dispatches()),
            ExecutionWork::default()
        );
    }

    #[test]
    fn overflowing_contributions_preserve_the_existing_lower_bound() {
        let known = ExecutionWork {
            matrix_flops: u64::MAX,
            largest_input_bytes: 7,
        };
        assert_eq!(
            known.including(ExecutionWork {
                matrix_flops: 1,
                largest_input_bytes: 33
            }),
            known
        );
    }

    #[test]
    fn missing_timestamps_are_observed_once() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = Arc::clone(&calls);
        let observation = Observation {
            work: ExecutionWork::default(),
            result: OnceLock::new(),
            validator: Arc::new(move |_, elapsed| {
                assert!(elapsed.is_none());
                observed.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }),
        };
        for _ in 0..2 {
            observation.check(None).unwrap();
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn metal_replay_reuses_execution_work_for_static_and_rebound_graphs() {
        let backend = MetalBackend::new()
            .unwrap()
            .with_execution_validator(Arc::new(|work, _| {
                assert_eq!(work.largest_input_bytes, 4097 * 4);
                Ok(())
            }));
        let input = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        let replacement = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        let output = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        let mut template =
            GraphTemplate::new(ParamSpace::new(vec![]).unwrap(), GraphLimits::default());
        for _ in 0..33 {
            template
                .dispatch(Op::Copy, &[&input.clone().into()], &output.clone().into())
                .unwrap();
        }
        let before = CALCULATIONS.get();
        let graph = backend.prepare_graph(template).unwrap();
        let prepared = CALCULATIONS.get();
        assert_eq!(prepared - before, 33);
        let mut bindings = forja_core::ReplayBindings::default();
        bindings.view(input, replacement).unwrap();
        for _ in 0..4 {
            backend.replay(&graph, vec![]).unwrap().wait().unwrap();
            backend
                .replay_bound(&graph, vec![], &bindings, false)
                .unwrap()
                .wait()
                .unwrap();
        }
        assert_eq!(CALCULATIONS.get(), prepared);
        eprintln!(
            "execution-work diagnostic (not a benchmark): 33 static dispatches; 8 ordinary/bound replays; 0 replay cost calculations"
        );
    }

    #[test]
    fn repeated_waits_preserve_a_validator_refusal() {
        let calls = Arc::new(AtomicUsize::new(0));
        let observed = calls.clone();
        let observation = Observation {
            work: ExecutionWork::default(),
            result: OnceLock::new(),
            validator: Arc::new(move |_, _| {
                observed.fetch_add(1, Ordering::Relaxed);
                Err(BackendError::ImpossibleRate {
                    resource: "FLOPs",
                    achieved: 111,
                    limit: 110,
                })
            }),
        };
        for _ in 0..2 {
            assert!(observation.check(Some(Duration::from_secs(1))).is_err());
        }
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
