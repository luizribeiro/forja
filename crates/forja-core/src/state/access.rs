use crate::{PagePoolError, StateSlot};

/// Authorized input and output identities for one state dispatch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StateStep {
    active: bool,
    source: u32,
    destination: u32,
}

impl StateStep {
    /// Resolves an active slot after validating both allocation generations.
    ///
    /// # Errors
    /// Refuses inactive, aliased or unauthenticated versions.
    pub fn from_slot(slot: &StateSlot) -> Result<Self, PagePoolError> {
        let access = StateAccess::from_slot(slot)?;
        if access.chain_len() != 1 {
            return Err(PagePoolError::InvalidCheckpoint);
        }
        access.step(0)
    }

    /// Prevents private data access for a padded lane regardless of device metadata.
    #[must_use]
    pub const fn inactive(mut self) -> Self {
        self.active = false;
        self
    }

    /// Whether the host authorizes private data access for this lane.
    #[must_use]
    pub const fn is_active(self) -> bool {
        self.active
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

/// Ordered version chain authorized by the trusted host for every state plane.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateAccess {
    active: bool,
    source: u32,
    preceding_destinations: Vec<u32>,
    final_destination: u32,
}

impl StateAccess {
    /// Builds inactive chain operands for poisoned metadata and graph tracing.
    ///
    /// The caller must provide backing for records `0..=output_count`.
    ///
    /// # Errors
    /// Refuses empty chains or chains beyond the padding slot capacity.
    pub fn poisoned(output_count: u32, output_capacity: u32) -> Result<Self, PagePoolError> {
        let slot = StateSlot::poisoned(output_count, output_capacity)?;
        let final_destination = u32::try_from(
            slot.table()
                .entries()
                .len()
                .checked_sub(1)
                .ok_or(PagePoolError::ArithmeticOverflow)?,
        )
        .map_err(|_| PagePoolError::ArithmeticOverflow)?;
        Ok(Self {
            active: false,
            source: 0,
            preceding_destinations: (1..final_destination).collect(),
            final_destination,
        })
    }

    /// Resolves every allocation generation and requires a nonempty disjoint chain.
    ///
    /// # Errors
    /// Refuses inactive, empty, aliased or unauthenticated chains.
    pub fn from_slot(slot: &StateSlot) -> Result<Self, PagePoolError> {
        if !slot.is_active() {
            return Err(PagePoolError::ForeignResource);
        }
        let mut records = (0..slot.table().entries().len())
            .map(|index| {
                u32::try_from(index)
                    .map_err(|_| PagePoolError::ArithmeticOverflow)
                    .and_then(|index| {
                        slot.table()
                            .resolve(index)
                            .map_err(|_| PagePoolError::CorruptState)
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let final_destination = records.pop().ok_or(PagePoolError::CorruptState)?;
        let Some((&source, destinations)) = records.split_first() else {
            return Err(PagePoolError::CorruptState);
        };
        let mut seen = std::collections::HashSet::from([source, final_destination]);
        if source == final_destination || destinations.iter().any(|record| !seen.insert(*record)) {
            return Err(PagePoolError::SharedTail);
        }
        Ok(Self {
            active: true,
            source,
            preceding_destinations: destinations.to_vec(),
            final_destination,
        })
    }

    /// Prevents private access for a padded lane regardless of device metadata.
    #[must_use]
    pub fn inactive(mut self) -> Self {
        self.active = false;
        self
    }

    /// Whether the host authorizes private access.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.active
    }

    /// Physical input identity before the first update.
    #[must_use]
    pub const fn source(&self) -> u32 {
        self.source
    }

    /// Physical output identity after the final update.
    #[must_use]
    pub const fn destination(&self) -> u32 {
        self.final_destination
    }

    /// Number of required updates per plane.
    #[must_use]
    pub fn chain_len(&self) -> usize {
        self.preceding_destinations.len() + 1
    }

    /// Ordered physical output identities.
    pub fn destinations(&self) -> impl Iterator<Item = u32> + '_ {
        self.preceding_destinations
            .iter()
            .copied()
            .chain(std::iter::once(self.final_destination))
    }

    /// Authorizes one dispatch using the immediately preceding version as its source.
    ///
    /// # Errors
    /// Refuses an update past the end of the chain.
    pub fn step(&self, index: usize) -> Result<StateStep, PagePoolError> {
        let destination = if index == self.preceding_destinations.len() {
            self.final_destination
        } else {
            *self
                .preceding_destinations
                .get(index)
                .ok_or(PagePoolError::InvalidCheckpoint)?
        };
        let source = if index == 0 {
            self.source
        } else {
            *self
                .preceding_destinations
                .get(index - 1)
                .ok_or(PagePoolError::InvalidCheckpoint)?
        };
        Ok(StateStep {
            active: self.active,
            source,
            destination,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poisoned_chains_have_only_inactive_steps() {
        for count in 1..=7 {
            let access = StateAccess::poisoned(count, 7).unwrap();
            assert_eq!(access.chain_len(), count as usize);
            for index in 0..count as usize {
                let step = access.step(index).unwrap();
                assert!(!step.is_active());
                assert_ne!(step.source(), step.destination());
            }
            assert!(access.step(count as usize).is_err());
        }
        assert!(StateAccess::poisoned(0, 7).is_err());
        assert!(StateAccess::poisoned(u32::MAX, 7).is_err());
    }
}
