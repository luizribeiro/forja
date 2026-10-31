use crate::{DType, PagedLayout, StateLayout};
use sha2::{Digest, Sha256};
use std::num::NonZeroU32;

/// Artifact namespace extended by cache geometry and canonical reduction choices.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct HybridPrefixNamespace {
    key: [u8; 32],
    state: [u8; 32],
    attention: [u8; 32],
    block: NonZeroU32,
}

impl HybridPrefixNamespace {
    /// Extends the existing weights/component/profile/tokenizer/chat namespace.
    ///
    /// # Errors
    /// Refuses empty, oversized or unnamed reduction-class declarations.
    pub fn new(
        base: [u8; 32],
        state: &StateLayout,
        attention: &PagedLayout,
        block: NonZeroU32,
        classes: &[&str],
    ) -> Result<Self, crate::BackendError> {
        if classes.is_empty()
            || classes.len() > 1024
            || classes
                .iter()
                .any(|name| name.is_empty() || name.len() > 64)
        {
            return Err(crate::BackendError::InvalidInput);
        }
        let state = state_hash(state);
        let attention = attention_hash(attention);
        let mut hash = Sha256::new();
        hash.update(b"forja-hybrid-prefix-v1");
        hash.update(base);
        hash.update(state);
        hash.update(attention);
        hash.update(block.get().to_le_bytes());
        for name in classes {
            hash.update(
                u64::try_from(name.len())
                    .map_err(|_| crate::BackendError::InvalidInput)?
                    .to_le_bytes(),
            );
            hash.update(name.as_bytes());
        }
        Ok(Self {
            key: hash.finalize().into(),
            state,
            attention,
            block,
        })
    }

    /// Canonical prompt block size declared by the engine.
    #[must_use]
    pub const fn block(self) -> NonZeroU32 {
        self.block
    }

    /// Checks state geometry before admitting a snapshot or attaching its backing.
    #[must_use]
    pub fn matches_state(self, layout: &StateLayout) -> bool {
        self.state == state_hash(layout)
    }

    /// Checks every attention plane and its page geometry.
    #[must_use]
    pub fn matches_attention(self, layout: &PagedLayout) -> bool {
        self.attention == attention_hash(layout)
    }
}

fn dtype_code(dtype: DType) -> u8 {
    match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::U32 => 3,
        DType::I32 => 4,
    }
}
fn state_hash(layout: &StateLayout) -> [u8; 32] {
    let mut hash = Sha256::new();
    for plane in layout.planes() {
        hash.update([dtype_code(plane.dtype())]);
        hash.update(plane.elements().to_le_bytes());
        hash.update((plane.shape().len() as u64).to_le_bytes());
        for dimension in plane.shape() {
            hash.update(dimension.to_le_bytes());
        }
    }
    hash.finalize().into()
}
fn attention_hash(layout: &PagedLayout) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(layout.page_tokens().to_le_bytes());
    for plane in layout.planes() {
        hash.update([dtype_code(plane.dtype)]);
        hash.update(plane.elements_per_token.to_le_bytes());
        hash.update(plane.alignment.to_le_bytes());
    }
    hash.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PagedPlane, StatePlane};
    #[test]
    fn namespaces_isolate_artifacts_geometry_blocks_and_reductions() {
        let state =
            StateLayout::new(vec![StatePlane::new(DType::BF16, vec![3, 33]).unwrap()]).unwrap();
        let pages = PagedLayout::new(
            64,
            vec![PagedPlane {
                dtype: DType::F16,
                elements_per_token: 128,
                alignment: 16,
                label: "key".into(),
            }],
        )
        .unwrap();
        let make = |base, block, class| {
            HybridPrefixNamespace::new(
                base,
                &state,
                &pages,
                NonZeroU32::new(block).unwrap(),
                &[class],
            )
            .unwrap()
        };
        let namespace = make([1; 32], 64, "fixed-tree");
        for other in [
            make([2; 32], 64, "fixed-tree"),
            make([1; 32], 32, "fixed-tree"),
            make([1; 32], 64, "serial"),
        ] {
            assert_ne!(namespace, other);
        }
        assert!(namespace.matches_state(&state) && namespace.matches_attention(&pages));
        let other =
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 33]).unwrap()]).unwrap();
        assert!(!namespace.matches_state(&other));
        assert!(HybridPrefixNamespace::new([1; 32], &state, &pages, NonZeroU32::MIN, &[]).is_err());
    }
}
