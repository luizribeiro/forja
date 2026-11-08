use std::{cell::RefCell, sync::Arc};
thread_local! { static CURRENT: RefCell<Option<Vec<usize>>> = const { RefCell::new(None) }; }

pub(super) struct Scope(Option<Vec<usize>>);
impl Scope {
    pub(super) fn enter() -> Self {
        Self(CURRENT.replace(Some(Vec::new())))
    }
    pub(super) fn finish(self) -> Arc<[usize]> {
        let mut keys = CURRENT.replace(None).unwrap_or_default();
        keys.sort_unstable();
        keys.dedup();
        drop(self);
        keys.into()
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.set(self.0.take());
    }
}
pub(super) fn record(key: usize) {
    CURRENT.with_borrow_mut(|keys| {
        if let Some(keys) = keys {
            keys.push(key);
        }
    });
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identities_are_canonical_and_nested_recording_restores_its_owner() {
        let outer = Scope::enter();
        record(7);
        record(1);
        let inner = Scope::enter();
        record(33);
        record(33);
        assert_eq!(&*inner.finish(), &[33]);
        record(7);
        assert_eq!(&*outer.finish(), &[1, 7]);
    }
}
