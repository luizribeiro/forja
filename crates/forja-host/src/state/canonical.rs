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
}
