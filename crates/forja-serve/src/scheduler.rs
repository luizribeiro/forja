use std::collections::{BTreeMap, VecDeque};

const MAX_BYPASSES: u8 = 8;
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct RequestId(pub(crate) u64);
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Admission {
    pub(crate) id: RequestId,
    pub(crate) reserved_tokens: u64,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Queued {
    admission: Admission,
    bypasses: u8,
}
pub(crate) struct SchedulerPolicy {
    queue: VecDeque<Queued>,
    decode: VecDeque<RequestId>,
    prompt: BTreeMap<RequestId, u32>,
    buckets: Vec<u32>,
    active_limit: usize,
    prompt_budget: u32,
    active: usize,
    submissions: u8,
}
impl SchedulerPolicy {
    pub(crate) fn new(
        mut buckets: Vec<u32>,
        active_limit: usize,
        prompt_budget: u32,
    ) -> Result<Self, &'static str> {
        buckets.sort_unstable();
        buckets.dedup();
        if buckets.first().copied() != Some(1)
            || buckets
                .last()
                .copied()
                .and_then(|value| usize::try_from(value).ok())
                != Some(active_limit)
        {
            return Err("decode buckets must span the active-lane ceiling");
        }
        if prompt_budget == 0 {
            return Err("prompt budget must be positive");
        }
        Ok(Self {
            queue: VecDeque::new(),
            decode: VecDeque::new(),
            prompt: BTreeMap::new(),
            buckets,
            active_limit,
            prompt_budget,
            active: 0,
            submissions: 0,
        })
    }
    pub(crate) fn enqueue(&mut self, admission: Admission) {
        self.queue.push_back(Queued {
            admission,
            bypasses: 0,
        });
    }
    pub(crate) fn cancel(&mut self, id: RequestId) -> bool {
        if let Some(index) = self
            .queue
            .iter()
            .position(|queued| queued.admission.id == id)
        {
            self.queue.remove(index);
            return true;
        }
        false
    }
    pub(crate) fn complete(&mut self, id: RequestId) {
        self.decode.retain(|candidate| *candidate != id);
        self.prompt.remove(&id);
        self.active = self.active.saturating_sub(1);
    }

    pub(crate) fn admit(&mut self, mut reserve: impl FnMut(Admission) -> bool) -> Vec<RequestId> {
        let mut admitted = Vec::new();
        while self.active < self.active_limit && !self.queue.is_empty() {
            let index = self.admission_candidate(&mut reserve);
            let Some(index) = index else {
                break;
            };
            let Some(queued) = self.queue.remove(index) else {
                break;
            };
            self.active += 1;
            admitted.push(queued.admission.id);
        }
        admitted
    }

    fn admission_candidate(
        &mut self,
        reserve: &mut impl FnMut(Admission) -> bool,
    ) -> Option<usize> {
        let head = self.queue.front()?.admission;
        if reserve(head) {
            return Some(0);
        }
        if self.queue.front()?.bypasses >= MAX_BYPASSES {
            return None;
        }
        let candidate = self
            .queue
            .iter()
            .enumerate()
            .skip(1)
            .find_map(|(index, queued)| {
                (queued.admission.reserved_tokens < head.reserved_tokens
                    && reserve(queued.admission))
                .then_some(index)
            });
        if candidate.is_some()
            && let Some(head) = self.queue.front_mut()
        {
            head.bypasses = head.bypasses.saturating_add(1);
        }
        candidate
    }

    pub(crate) fn ready_decode(&mut self, id: RequestId) {
        if !self.decode.contains(&id) {
            self.decode.push_back(id);
        }
    }

    pub(crate) fn ready_prompt(&mut self, id: RequestId, tokens: u32) {
        self.prompt.insert(id, tokens);
    }

    pub(crate) fn decode_batches(&mut self) -> Vec<(u32, Vec<RequestId>)> {
        let mut batches = Vec::new();
        while !self.decode.is_empty() {
            let active = self.decode.len().min(self.active_limit);
            let bucket = self
                .buckets
                .iter()
                .copied()
                .find(|&bucket| usize::try_from(bucket).is_ok_and(|bucket| bucket >= active))
                .unwrap_or_else(|| self.buckets[self.buckets.len() - 1]);
            let count = active.min(usize::try_from(bucket).unwrap_or(usize::MAX));
            let lanes = self.decode.drain(..count).collect();
            batches.push((bucket, lanes));
        }
        batches
    }

    pub(crate) fn prompt_batch(&mut self) -> Vec<(RequestId, u32)> {
        let mut remaining = self.prompt_budget;
        let ids = self.prompt.keys().copied().collect::<Vec<_>>();
        let mut batch = Vec::new();
        for id in ids {
            let Some(tokens) = self.prompt.get(&id).copied() else {
                continue;
            };
            let spend = tokens.min(remaining);
            if spend == 0 {
                break;
            }
            batch.push((id, spend));
            remaining -= spend;
            if spend == tokens {
                self.prompt.remove(&id);
            } else {
                self.prompt.insert(id, tokens - spend);
            }
        }
        batch
    }

    pub(crate) fn can_submit(&self) -> bool {
        self.submissions < 2
    }

    pub(crate) fn submitted(&mut self) -> Result<(), &'static str> {
        if !self.can_submit() {
            return Err("scheduler pipeline is full");
        }
        self.submissions += 1;
        Ok(())
    }

    pub(crate) fn completed_submission(&mut self) {
        self.submissions = self.submissions.saturating_sub(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn admits_fifo_with_bounded_smaller_request_bypass() {
        let mut policy = SchedulerPolicy::new(vec![1, 2, 4, 8], 8, 2_048).unwrap();
        policy.enqueue(Admission {
            id: RequestId(1),
            reserved_tokens: 4_096,
        });
        policy.enqueue(Admission {
            id: RequestId(2),
            reserved_tokens: 64,
        });
        assert_eq!(
            policy.admit(|request| request.reserved_tokens < 1_000),
            [RequestId(2)]
        );
        for id in 3..=10 {
            policy.enqueue(Admission {
                id: RequestId(id),
                reserved_tokens: 64,
            });
        }
        assert_eq!(
            policy
                .admit(|request| request.reserved_tokens < 1_000)
                .len(),
            7
        );
        for id in 2..=9 {
            policy.complete(RequestId(id));
        }
        assert_eq!(policy.admit(|request| request.reserved_tokens < 1_000), []);
    }

    #[test]
    fn prioritizes_decode_and_caps_prompt_work_and_pipeline_depth() {
        let mut policy = SchedulerPolicy::new(vec![1, 2, 4, 8], 8, 2_048).unwrap();
        for id in 1..=5 {
            policy.ready_decode(RequestId(id));
        }
        assert_eq!(policy.decode_batches()[0].0, 8);
        policy.ready_prompt(RequestId(7), 2_000);
        policy.ready_prompt(RequestId(8), 1_000);
        assert_eq!(
            policy.prompt_batch(),
            [(RequestId(7), 2_000), (RequestId(8), 48)]
        );
        policy.submitted().unwrap();
        policy.submitted().unwrap();
        assert!(policy.submitted().is_err());
        policy.completed_submission();
        assert!(policy.can_submit());
    }
}
