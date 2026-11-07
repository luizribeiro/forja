use crate::{Dispatch, Op};
use std::collections::{HashMap, VecDeque};

const SHAPES_PER_DISPATCH: usize = 8;

#[derive(Debug, Default)]
pub(super) struct WorkCache {
    values: HashMap<usize, VecDeque<(Vec<u64>, u64)>>,
    #[cfg(test)]
    pub(super) computations: usize,
}

impl WorkCache {
    pub(super) fn work(&mut self, index: usize, dispatch: &Dispatch) -> Option<u64> {
        let mut shape = Vec::new();
        for tensor in dispatch.inputs().iter().chain(dispatch.outputs()) {
            shape.push(u64::try_from(tensor.layout().shape().len()).ok()?);
            shape.extend(tensor.layout().shape().iter().map(|&n| u64::from(n)));
        }
        match dispatch.op() {
            Op::Sdpa {
                causal: true,
                q_start,
                ..
            }
            | Op::PagedSdpaAppend { q_start, .. } => shape.push(u64::from(q_start)),
            Op::PagedSdpa {
                q_start,
                key_length,
                ..
            } => shape.extend([u64::from(q_start), u64::from(key_length)]),
            _ => {}
        }
        let shapes = self.values.entry(index).or_default();
        if let Some((_, work)) = shapes.iter().find(|(key, _)| key == &shape) {
            return Some(*work);
        }
        let work = dispatch.work()?;
        #[cfg(test)]
        {
            self.computations += 1;
        }
        if shapes.len() == SHAPES_PER_DISPATCH {
            shapes.pop_front();
        }
        shapes.push_back((shape, work));
        Some(work)
    }
}
