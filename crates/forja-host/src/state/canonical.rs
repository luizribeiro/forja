use super::StateCharge;
use crate::{Host, compute, quota};
use forja_core::{Backend, HybridPrefixNamespace};
use std::sync::Arc;

/// Exact prompt ledger authorized for canonical block evaluation in one host.
#[derive(Debug)]
pub struct CanonicalPrompt {
    pub(super) namespace: HybridPrefixNamespace,
    pub(super) tokens: Vec<u32>,
    pub(super) charge: Arc<StateCharge>,
}

impl CanonicalPrompt {
    /// Namespace covering the artifacts, cache layouts and numerical classes.
    #[must_use]
    pub const fn namespace(&self) -> HybridPrefixNamespace {
        self.namespace
    }
    /// Exact input tokens, including the final token that prefix reuse must replay.
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }
    /// Charged ledger and allocation metadata retained by all shared references.
    #[must_use]
    pub fn charged_bytes(&self) -> u64 {
        self.charge.bytes
    }
}

impl<B: Backend + 'static> Host<B> {
    /// Charges an immutable token ledger without admitting a state update.
    ///
    /// # Errors
    /// Refuses unusable prompts, arithmetic overflow or the existing host byte quota.
    pub fn canonical_prompt(
        &self,
        namespace: HybridPrefixNamespace,
        tokens: Vec<u32>,
    ) -> Result<Arc<CanonicalPrompt>, compute::Error> {
        let count = u64::try_from(tokens.len()).map_err(|_| quota("prompt length overflow"))?;
        if count < 2 || count > self.limits.tensor_elements {
            return Err(quota("canonical prompt exceeds token limits"));
        }
        let bytes = u64::try_from(tokens.capacity())
            .map_err(|_| quota("prompt capacity overflow"))?
            .checked_mul(4)
            .and_then(|n| n.checked_add(256))
            .ok_or_else(|| quota("prompt ledger bytes overflow"))?;
        let charge = StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.live_bytes),
            self.limits.live_bytes,
            bytes,
        )?;
        Ok(Arc::new(CanonicalPrompt {
            namespace,
            tokens,
            charge: Arc::new(charge),
        }))
    }
}

#[derive(Clone, Default)]
pub(super) enum CanonicalPosition {
    #[default]
    Empty,
    Recurrent,
    Prefill(Arc<CanonicalPrompt>, u64),
}

impl CanonicalPosition {
    pub(super) fn permits(&self, prompt: &CanonicalPrompt, start: u64, end: u64) -> bool {
        let Ok(end_index) = usize::try_from(end) else {
            return false;
        };
        let Ok(start_index) = usize::try_from(start) else {
            return false;
        };
        let block = u64::from(prompt.namespace.block().get());
        if end <= start
            || end_index >= prompt.tokens.len()
            || !start.is_multiple_of(block)
            || !end.is_multiple_of(block)
        {
            return false;
        }
        match self {
            Self::Empty => start == 0,
            Self::Prefill(previous, through) => {
                *through == start
                    && previous.namespace == prompt.namespace
                    && previous
                        .tokens
                        .get(..start_index)
                        .is_some_and(|tokens| tokens == &prompt.tokens[..start_index])
            }
            Self::Recurrent => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use forja_core::{DType, PagedLayout, PagedPlane, StateLayout, StatePlane};
    use std::{num::NonZeroU32, sync::atomic::Ordering};
    pub(super) fn namespace() -> HybridPrefixNamespace {
        HybridPrefixNamespace::new(
            [7; 32],
            &StateLayout::new(vec![StatePlane::new(DType::F32, vec![1, 7, 33]).unwrap()]).unwrap(),
            &PagedLayout::new(
                64,
                vec![PagedPlane {
                    dtype: DType::F16,
                    elements_per_token: 33,
                    alignment: 2,
                    label: "key".into(),
                }],
            )
            .unwrap(),
            NonZeroU32::new(64).unwrap(),
            &["fixed-tree"],
        )
        .unwrap()
    }
    #[test]
    fn ledger_charge_is_shared_and_quota_refusal_is_atomic() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(1024, 8, 4096, 32, 4096),
        );
        assert!(host.canonical_prompt(namespace(), vec![1]).is_err());
        let mut oversized = Vec::with_capacity(1024);
        oversized.extend([1, 2]);
        assert!(host.canonical_prompt(namespace(), oversized).is_err());
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
        let prompt = host
            .canonical_prompt(namespace(), (0..129).collect())
            .unwrap();
        assert_eq!(prompt.tokens(), &(0..129).collect::<Vec<_>>());
        assert_eq!(prompt.namespace(), namespace());
        let bytes = host.live_bytes.load(Ordering::Acquire);
        assert_eq!(prompt.charged_bytes(), bytes);
        let shared = Arc::clone(&prompt);
        assert!(
            host.canonical_prompt(namespace(), (0..129).collect())
                .is_err()
        );
        assert_eq!(host.live_bytes.load(Ordering::Acquire), bytes);
        drop(prompt);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), bytes);
        drop(shared);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
    }
    #[test]
    fn origin_follows_publication_cancellation_and_rewind() {
        let host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(131_072, 8, 4096, 32, 4096),
        );
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![1, 7, 33]).unwrap()])
                    .unwrap(),
                1,
            )
            .unwrap();
        let sequence = storage.reserve().unwrap();
        assert!(sequence.has_pristine_state().unwrap());
        let pending = sequence.begin(1).unwrap();
        assert_eq!(
            sequence.has_pristine_state(),
            Err(forja_core::PagePoolError::BankInFlight)
        );
        sequence.cancel(&pending).unwrap();
        assert!(sequence.has_pristine_state().unwrap());
        let pending = sequence.begin(2).unwrap();
        sequence.lock().mark_submitted(&pending).unwrap();
        storage.pool.observe_completion(2).unwrap();
        sequence.lock().finish(&pending, true).unwrap();
        assert!(!sequence.has_pristine_state().unwrap());
        sequence.rewind(0).unwrap();
        assert!(sequence.has_pristine_state().unwrap());
    }
    #[test]
    fn canonical_blocks_require_exact_prompt_lineage_and_leave_a_tail() {
        for length in [63, 64, 65, 129] {
            let host = Host::new(
                forja_cpu::CpuBackend::new(),
                Limits::new(131_072, 8, 4096, 32, 4096),
            );
            let storage = host
                .create_state_storage(
                    StateLayout::new(vec![StatePlane::new(DType::F32, vec![1, 7, 33]).unwrap()])
                        .unwrap(),
                    1,
                )
                .unwrap();
            let sequence = storage.reserve().unwrap();
            let prompt = host
                .canonical_prompt(namespace(), (0..length).collect())
                .unwrap();
            let first = sequence.begin_prefill(
                1,
                Arc::clone(&prompt),
                64,
                257,
                super::super::StateAction::Step,
            );
            if length <= 64 {
                assert!(first.is_err());
                assert!(sequence.has_pristine_state().unwrap());
                continue;
            }
            let first = first.unwrap();
            assert!(sequence.canonical_token_count().is_err());
            sequence.lock().mark_submitted(&first).unwrap();
            storage.pool.observe_completion(1).unwrap();
            sequence.lock().finish(&first, true).unwrap();
            assert_eq!(sequence.canonical_token_count().unwrap(), Some(64));
            if length == 129 {
                let mut changed = prompt.tokens().to_vec();
                changed[33] = 999;
                let changed = host.canonical_prompt(namespace(), changed).unwrap();
                assert!(
                    sequence
                        .begin_prefill(2, changed, 128, 257, super::super::StateAction::Step)
                        .is_err()
                );
                let second = sequence
                    .begin_prefill(
                        2,
                        Arc::clone(&prompt),
                        128,
                        257,
                        super::super::StateAction::Step,
                    )
                    .unwrap();
                sequence.cancel(&second).unwrap();
                assert_eq!(sequence.canonical_token_count().unwrap(), Some(64));
            }
            let generated = sequence
                .begin_at(
                    3,
                    super::super::LanePosition {
                        tokens: 128,
                        reserved_tokens: 257,
                        ..sequence.position()
                    },
                )
                .unwrap();
            sequence.lock().mark_submitted(&generated).unwrap();
            storage.pool.observe_completion(3).unwrap();
            sequence.lock().finish(&generated, true).unwrap();
            assert_eq!(sequence.canonical_token_count().unwrap(), None);
            assert!(
                sequence
                    .begin_prefill(4, prompt, 192, 257, super::super::StateAction::Step)
                    .is_err()
            );
            sequence.rewind(64).unwrap();
            assert_eq!(sequence.canonical_token_count().unwrap(), Some(64));
        }
    }
}
