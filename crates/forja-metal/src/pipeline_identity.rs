use forja_core::{BackendError, program::ProgramHash};
use std::{
    hash::{DefaultHasher, Hash, Hasher},
    sync::Arc,
};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum MathMode {
    Fast,
    Relaxed,
    Safe,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct SourceIdentity {
    text: Arc<str>,
    mode: MathMode,
    fingerprint: u64,
}
impl SourceIdentity {
    pub(super) fn new(text: Arc<str>, mode: MathMode) -> Self {
        let mut hash = DefaultHasher::new();
        text.hash(&mut hash);
        mode.hash(&mut hash);
        Self {
            text,
            mode,
            fingerprint: hash.finish(),
        }
    }
}
impl Hash for SourceIdentity {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.fingerprint.hash(state);
    }
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct PipelineKey<S> {
    pub(super) source: S,
    pub(super) name: String,
    pub(super) constants: Vec<(u32, u32)>,
    hooks: Vec<ProgramHash>,
}
impl<S> PipelineKey<S> {
    pub(super) fn new(
        source: S,
        name: &str,
        constants: &[(u32, u32)],
        hooks: &[ProgramHash],
    ) -> Result<Self, BackendError> {
        Ok(Self {
            source,
            name: name.to_owned(),
            constants: crate::pipeline_constants::canonical(constants)?,
            hooks: hooks.to_vec(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keys_bind_full_source_mode_and_each_canonical_constant() {
        let source = SourceIdentity::new(Arc::from("kernel source"), MathMode::Fast);
        let key = PipelineKey::new(source.clone(), "entry", &[(7, 33), (1, 4097)], &[]).unwrap();
        assert_eq!(
            key,
            PipelineKey::new(source, "entry", &[(1, 4097), (7, 33)], &[]).unwrap()
        );
        for (text, mode) in [
            ("other source", MathMode::Fast),
            ("kernel source", MathMode::Safe),
            ("kernel source", MathMode::Relaxed),
        ] {
            assert_ne!(
                key,
                PipelineKey::new(
                    SourceIdentity::new(Arc::from(text), mode),
                    "entry",
                    &[(1, 4097), (7, 33)],
                    &[]
                )
                .unwrap()
            );
        }
        let hook = forja_core::program::Program {
            kind: forja_core::program::ProgramKind::Map,
            insts: vec![forja_core::program::Inst::Input(0)],
            outputs: vec![(0, 0)],
        }
        .validate()
        .unwrap()
        .content_hash();
        assert_ne!(
            key,
            PipelineKey::new(key.source.clone(), "entry", &key.constants, &[hook]).unwrap()
        );
        let mut collision = key.source.clone();
        collision.text = Arc::from("different source with the same bucket hash");
        assert_ne!(key.source, collision);
        assert!(PipelineKey::new(key.source.clone(), "entry", &[(1, 7), (1, 33)], &[]).is_err());
        assert_ne!(
            key,
            PipelineKey::new(key.source.clone(), "other entry", &key.constants, &[]).unwrap()
        );
        assert_ne!(
            key,
            PipelineKey::new(key.source.clone(), "entry", &[(1, 4097), (7, 34)], &[]).unwrap()
        );
    }
}
