use super::{BackendError, Dispatch, Duration, Op, launch_work::LaunchWork};
use crate::variants::MetalVariant;
use forja_core::{Layout, WorkCalibration};
use std::sync::{Arc, Mutex};

struct Entry {
    rates: Arc<WorkCalibration>,
    op: Op,
    variant: Option<MetalVariant>,
    work: LaunchWork,
    inputs: usize,
    layouts: Vec<Layout>,
    prediction: Duration,
}
impl Entry {
    fn matches(&self, rates: &Arc<WorkCalibration>, dispatch: &Dispatch, work: LaunchWork) -> bool {
        Arc::ptr_eq(&self.rates, rates)
            && same_cost_op(self.op, dispatch.op())
            && self.variant == super::selected_variant(dispatch)
            && self.work == work
            && self.inputs == dispatch.inputs().len()
            && self.layouts.len() == dispatch.inputs().len() + dispatch.outputs().len()
            && self
                .layouts
                .iter()
                .zip(
                    dispatch
                        .inputs()
                        .iter()
                        .chain(dispatch.outputs())
                        .map(forja_core::Tensor::layout),
                )
                .all(|(a, b)| same_cost_layout(a, b))
    }
}

// Binding validation owns allocation bounds; work counts use geometry and vector alignment.
fn same_cost_layout(a: &Layout, b: &Layout) -> bool {
    a.dtype() == b.dtype()
        && a.shape() == b.shape()
        && a.strides() == b.strides()
        && forja_core::ByteAlignment::of(a.byte_span().start)
            == forja_core::ByteAlignment::of(b.byte_span().start)
}
fn same_cost_op(a: Op, b: Op) -> bool {
    match (a, b) {
        // Authentication identities select protected storage, not the stage schedule.
        (Op::State(a), Op::State(b)) => std::mem::discriminant(&a) == std::mem::discriminant(&b),
        _ => a == b,
    }
}

// A cache belongs to one immutable graph node, including its program and epilogue.
#[derive(Default)]
pub(super) struct NodeCache {
    entries: Mutex<Vec<Entry>>,
    #[cfg(test)]
    pub(super) calculations: std::sync::atomic::AtomicUsize,
}
impl NodeCache {
    pub(super) fn predict(
        &self,
        rates: &Arc<WorkCalibration>,
        dispatch: &Dispatch,
        work: LaunchWork,
        calculate: impl FnOnce() -> Result<Duration, BackendError>,
    ) -> Result<Duration, BackendError> {
        let mut entries = self
            .entries
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        if let Some(entry) = entries
            .iter()
            .find(|entry| entry.matches(rates, dispatch, work))
        {
            return Ok(entry.prediction);
        }
        let prediction = calculate()?;
        #[cfg(test)]
        self.calculations
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let entry = Entry {
            rates: Arc::clone(rates),
            op: dispatch.op(),
            variant: super::selected_variant(dispatch),
            work,
            inputs: dispatch.inputs().len(),
            layouts: dispatch
                .inputs()
                .iter()
                .chain(dispatch.outputs())
                .map(|t| t.layout().clone())
                .collect(),
            prediction,
        };
        if entries.len() == 2 {
            entries.remove(0);
        }
        entries.push(entry);
        Ok(prediction)
    }
}

pub(super) fn split_nodes(
    nodes: &[Arc<NodeCache>],
    predictions: &[Duration],
    budget: Duration,
) -> Result<Vec<Arc<NodeCache>>, BackendError> {
    if nodes.len() != predictions.len() {
        return Err(BackendError::InvalidInput);
    }
    Ok(nodes
        .iter()
        .zip(predictions)
        .flat_map(|(cache, &time)| {
            if time > budget {
                vec![Arc::default(), Arc::default()]
            } else {
                vec![Arc::clone(cache)]
            }
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{CommandList, DType};
    use std::cell::Cell;

    fn copy(cpu: &forja_cpu::CpuBackend, rows: u32) -> Dispatch {
        let input = cpu.alloc(DType::F32, &[rows, 33]).unwrap();
        let output = cpu.alloc(DType::F32, &[rows, 33]).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();
        commands.into_dispatches().remove(0)
    }
    #[test]
    fn aligned_offsets_reuse_work_while_changed_alignment_is_repriced() {
        use forja_core::{Slice, ViewOp};
        let _rates = crate::work_admission::TestCalibration::fast();
        let backend = crate::MetalBackend::new().unwrap();
        let rates = crate::work_admission::calibration(&backend).unwrap();
        let cpu = forja_cpu::CpuBackend::new();
        let cache = NodeCache::default();
        let initial = copy(&cpu, 7);
        let calculate = || Ok(Duration::from_nanos(1));
        let work = LaunchWork {
            groups: 1,
            threads: 128,
            launches: 1,
            attention: None,
        };
        cache.predict(&rates, &initial, work, calculate).unwrap();
        for (offset, expected) in [(4, 1), (1, 2)] {
            let storage = cpu.alloc(DType::F32, &[7 * 33 + 4]).unwrap();
            let view = cpu
                .view(
                    &storage,
                    ViewOp::Slice(vec![Slice::new(offset, 7 * 33, 1).unwrap()]),
                )
                .unwrap();
            let input = cpu.view(&view, ViewOp::Reshape(vec![7, 33])).unwrap();
            let output = cpu.alloc(DType::F32, &[7, 33]).unwrap();
            let mut commands = CommandList::new();
            commands.dispatch(Op::Copy, &[&input], &output).unwrap();
            cache
                .predict(&rates, commands.last_dispatch().unwrap(), work, calculate)
                .unwrap();
            assert_eq!(
                cache
                    .calculations
                    .load(std::sync::atomic::Ordering::Relaxed),
                expected
            );
        }
        let access = forja_testing::state_test_metadata().unwrap().0;
        assert!(same_cost_op(
            Op::State(forja_core::StateOp::DeltaRule(access)),
            Op::State(forja_core::StateOp::DeltaRule(access.inactive()))
        ));
        assert!(!same_cost_op(
            Op::State(forja_core::StateOp::DeltaRule(access)),
            Op::State(forja_core::StateOp::DeltaRuleBlock(access))
        ));
    }

    #[test]
    fn cache_reuses_shapes_across_buffers_and_binds_every_forecast_input() {
        let _rates = crate::work_admission::TestCalibration::fast();
        let backend = crate::MetalBackend::new().unwrap();
        let rates = crate::work_admission::calibration(&backend).unwrap();
        let cpu = forja_cpu::CpuBackend::new();
        let first = copy(&cpu, 7);
        let second = copy(&cpu, 7);
        assert_ne!(first.inputs()[0].buffer(), second.inputs()[0].buffer());
        let cache = NodeCache::default();
        let calls = Cell::new(0_u64);
        let calculate = || {
            calls.set(calls.get() + 1);
            Ok(Duration::from_nanos(calls.get()))
        };
        let work = LaunchWork {
            groups: 1,
            threads: 128,
            launches: 1,
            attention: None,
        };
        assert_eq!(
            cache.predict(&rates, &first, work, calculate).unwrap(),
            Duration::from_nanos(1)
        );
        assert_eq!(
            cache.predict(&rates, &second, work, calculate).unwrap(),
            Duration::from_nanos(1)
        );
        assert_eq!(calls.get(), 1);
        for change in [0, 1, 2, 3, 4, 5] {
            {
                let mut entries = cache.entries.lock().unwrap();
                let entry = entries.first_mut().unwrap();
                match change {
                    0 => entry.rates = Arc::new((*rates).clone()),
                    1 => entry.op = Op::Rope { theta: 1.0 },
                    2 => entry.variant = Some(MetalVariant::QuantMatmulTiled),
                    3 => entry.work.threads += 1,
                    4 => entry.work.attention = Some(super::super::AttentionSchedule::Decomposed),
                    _ => entry.layouts[0] = copy(&cpu, 33).inputs()[0].layout().clone(),
                }
            }
            let before = calls.get();
            cache.predict(&rates, &first, work, calculate).unwrap();
            assert_eq!(calls.get(), before + 1);
            cache.entries.lock().unwrap().remove(0);
        }
        for rows in [1, 33, 7] {
            cache
                .predict(&rates, &copy(&cpu, rows), work, calculate)
                .unwrap();
            assert!(cache.entries.lock().unwrap().len() <= 2);
        }
        let before = calls.get();
        cache
            .predict(&rates, &copy(&cpu, 1), work, calculate)
            .unwrap();
        assert_eq!(calls.get(), before + 1);
        assert_eq!(cache.entries.lock().unwrap().len(), 2);
    }
}
