use crate::ExecutionWork;
use forja_core::{CommandList, GraphTemplate};
use std::{
    collections::BTreeMap,
    ops::Range,
    sync::{Arc, Mutex},
    time::Duration,
};

#[derive(Clone, Eq, Ord, PartialEq, PartialOrd)]
enum Shape {
    Window(usize, usize),
    Split(Vec<bool>),
}

pub(crate) struct ExecutionPlan {
    pub(crate) total: ExecutionWork,
    dispatches: Vec<ExecutionWork>,
    shapes: Mutex<BTreeMap<Shape, Arc<Self>>>,
}

impl ExecutionPlan {
    pub(crate) fn graph(graph: &GraphTemplate) -> Arc<Self> {
        let mut work = vec![ExecutionWork::default(); graph.len()];
        for (index, dispatch) in graph.static_dispatches() {
            work[index] = ExecutionWork::of(std::slice::from_ref(dispatch));
        }
        Self::new(work)
    }

    fn new(dispatches: Vec<ExecutionWork>) -> Arc<Self> {
        Arc::new(Self {
            total: dispatches
                .iter()
                .copied()
                .fold(ExecutionWork::default(), ExecutionWork::including),
            dispatches,
            shapes: Mutex::default(),
        })
    }

    fn cached(&self, shape: Shape, build: impl FnOnce() -> Vec<ExecutionWork>) -> Arc<Self> {
        let mut shapes = self
            .shapes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(shapes.entry(shape).or_insert_with(|| Self::new(build())))
    }

    pub(crate) fn window(&self, range: Range<usize>) -> Arc<Self> {
        self.cached(Shape::Window(range.start, range.end), || {
            self.dispatches.get(range).unwrap_or_default().to_vec()
        })
    }

    pub(crate) fn split(
        &self,
        commands: &CommandList,
        predictions: &[Duration],
        budget: Duration,
    ) -> Arc<Self> {
        let divided = predictions
            .iter()
            .map(|&time| time > budget)
            .collect::<Vec<_>>();
        self.cached(Shape::Split(divided.clone()), || {
            let mut next = 0;
            self.dispatches
                .iter()
                .zip(divided)
                .flat_map(|(&work, divided)| {
                    let count = if divided { 2 } else { 1 };
                    let start = next;
                    next += count;
                    (start..next).map(move |index| {
                        if work == ExecutionWork::default() {
                            return work;
                        }
                        if !divided {
                            return work;
                        }
                        commands
                            .dispatches()
                            .get(index)
                            .map_or_else(ExecutionWork::default, |dispatch| {
                                ExecutionWork::of(std::slice::from_ref(dispatch))
                            })
                    })
                })
                .collect()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{DType, GraphLimits, Op, ParamSpace};

    #[test]
    fn repeated_windows_and_output_splits_share_their_shape_summary() {
        let backend = forja_cpu::CpuBackend::new();
        let input = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let output = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let mut template =
            GraphTemplate::new(ParamSpace::new(vec![]).unwrap(), GraphLimits::default());
        template
            .dispatch(Op::Copy, &[&input.into()], &output.into())
            .unwrap();
        let plan = ExecutionPlan::graph(&template);
        let window = plan.window(0..1);
        assert!(Arc::ptr_eq(&window, &plan.window(0..1)));
        let commands = template
            .instantiate(&template.values(vec![]).unwrap())
            .unwrap();
        let halves = commands.dispatches()[0]
            .split_output_rows()
            .unwrap()
            .unwrap();
        let mut split = CommandList::new();
        for half in halves {
            split.record_validated(half);
        }
        let times = [Duration::from_secs(1)];
        let divided = plan.split(&split, &times, Duration::ZERO);
        assert!(Arc::ptr_eq(
            &divided,
            &plan.split(&split, &times, Duration::ZERO)
        ));
        assert_eq!(divided.total, ExecutionWork::of(split.dispatches()));
    }
}
