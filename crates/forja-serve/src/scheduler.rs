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
    prompt: BTreeMap<RequestId, u32>,
    buckets: Vec<u32>,
    active_limit: usize,
    prompt_budget: u32,
    active: usize,
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
                .is_none_or(|largest| largest < active_limit)
        {
            return Err("decode buckets must span the active-lane ceiling");
        }
        if prompt_budget == 0 {
            return Err("prompt budget must be positive");
        }
        Ok(Self {
            queue: VecDeque::new(),
            prompt: BTreeMap::new(),
            buckets,
            active_limit,
            prompt_budget,
            active: 0,
        })
    }
    pub(crate) fn enqueue(&mut self, admission: Admission) {
        self.queue.push_back(Queued {
            admission,
            bypasses: 0,
        });
    }

    pub(crate) fn reconcile_active(&mut self, active: usize) {
        self.active = active.min(self.active_limit);
    }

    pub(crate) fn decode_bucket(&self, slots: impl IntoIterator<Item = u32>) -> Option<u32> {
        let needed = slots.into_iter().map(|slot| slot.saturating_add(1)).max()?;
        self.buckets
            .iter()
            .copied()
            .find(|&bucket| bucket >= needed)
    }
    pub(crate) fn complete(&mut self, id: RequestId) {
        self.prompt.remove(&id);
        self.active = self.active.saturating_sub(1);
    }

    pub(crate) fn retry_admission(&mut self, admission: Admission) {
        self.active = self.active.saturating_sub(1);
        self.queue.push_front(Queued {
            admission,
            bypasses: MAX_BYPASSES,
        });
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

    pub(crate) fn ready_prompt(&mut self, id: RequestId, tokens: u32) {
        self.prompt.insert(id, tokens);
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
    fn caps_prompt_work_and_selects_decode_buckets() {
        let mut policy = SchedulerPolicy::new(vec![1, 2, 4, 8], 8, 2_048).unwrap();
        policy.ready_prompt(RequestId(7), 2_000);
        policy.ready_prompt(RequestId(8), 1_000);
        assert_eq!(
            policy.prompt_batch(),
            [(RequestId(7), 2_000), (RequestId(8), 48)]
        );
        assert_eq!(policy.decode_bucket([0, 2]), Some(4));
    }

    #[test]
    fn transient_admission_retry_keeps_the_request_at_the_head() {
        let mut policy = SchedulerPolicy::new(vec![1, 2], 2, 16).unwrap();
        let first = Admission {
            id: RequestId(1),
            reserved_tokens: 8,
        };
        policy.enqueue(first);
        policy.enqueue(Admission {
            id: RequestId(2),
            reserved_tokens: 4,
        });
        assert_eq!(policy.admit(|_| true), [RequestId(1), RequestId(2)]);
        policy.retry_admission(first);
        policy.complete(RequestId(2));
        assert_eq!(policy.admit(|_| true), [RequestId(1)]);
    }
}
