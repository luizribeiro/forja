use crate::{PagePoolError, StateSlot};

/// Authorized input and output version identities supplied only by the trusted host.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateAccess {
    source: u32,
    destination: u32,
}

impl StateAccess {
    /// Resolves an active slot after validating both allocation generations.
    ///
    /// # Errors
    /// Refuses inactive, aliased or unauthenticated versions.
    pub fn from_slot(slot: &StateSlot) -> Result<Self, PagePoolError> {
        if !slot.is_active() {
            return Err(PagePoolError::ForeignResource);
        }
        let source = slot
            .table()
            .resolve(0)
            .map_err(|_| PagePoolError::CorruptState)?;
        let destination = slot
            .table()
            .resolve(1)
            .map_err(|_| PagePoolError::CorruptState)?;
        if source == destination {
            return Err(PagePoolError::SharedTail);
        }
        Ok(Self {
            source,
            destination,
        })
    }

    /// Expected physical input identity for backend authentication, never a guest address.
    #[must_use]
    pub const fn source(self) -> u32 {
        self.source
    }

    /// Expected private output identity for backend authentication.
    #[must_use]
    pub const fn destination(self) -> u32 {
        self.destination
    }
}
