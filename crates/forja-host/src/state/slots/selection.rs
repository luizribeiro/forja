use super::super::{StateSelection, StateSelectionLease};
use crate::{compute, invalid_handle};
use forja_core::Backend;
use std::sync::{Mutex, Weak};

pub(super) enum SelectionFrame<B: Backend> {
    Owned(StateSelection<B>),
    Shared(Weak<Mutex<Option<StateSelection<B>>>>),
}

impl<B: Backend> SelectionFrame<B> {
    pub(super) fn with<T>(
        &self,
        action: impl FnOnce(&StateSelection<B>) -> Result<T, compute::Error>,
    ) -> Result<Option<T>, compute::Error> {
        match self {
            Self::Owned(selection) => action(selection).map(Some),
            Self::Shared(weak) => {
                let Some(owner) = weak.upgrade() else {
                    return Ok(None);
                };
                let guard = owner
                    .lock()
                    .map_err(|_| invalid_handle("selection ownership is poisoned"))?;
                guard.as_ref().map(action).transpose()
            }
        }
    }

    pub(super) fn owner(&self) -> Result<Option<StateSelectionLease<B>>, compute::Error> {
        match self {
            Self::Owned(_) => Ok(None),
            Self::Shared(weak) => weak
                .upgrade()
                .map(StateSelectionLease::borrow)
                .transpose()
                .map(Option::flatten),
        }
    }

    pub(super) fn take(self) -> Result<Option<StateSelection<B>>, compute::Error> {
        match self {
            Self::Owned(selection) => Ok(Some(selection)),
            Self::Shared(weak) => {
                let Some(owner) = weak.upgrade() else {
                    return Ok(None);
                };
                owner
                    .lock()
                    .map(|mut guard| guard.take())
                    .map_err(|_| invalid_handle("selection ownership is poisoned"))
            }
        }
    }
}
