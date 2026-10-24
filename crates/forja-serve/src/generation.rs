//! Runtime-generation resource accounting.

/// A no-allocation plan for a generation's fully retained memory.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadPlan {
    /// Exact bytes retained after startup, when proven from an equivalent load.
    pub retained_bytes: Option<u64>,
}

/// Shared memory reservations for all resident runtime generations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcessLedger {
    limit: u64,
    reserved: u64,
}

impl ProcessLedger {
    /// Creates an empty ledger with the process-wide byte limit.
    #[must_use]
    pub const fn new(limit: u64) -> Self {
        Self { limit, reserved: 0 }
    }

    /// Reserves bytes without exceeding the process-wide limit.
    ///
    /// # Errors
    ///
    /// Returns an error if addition overflows or the reservation exceeds the limit.
    pub fn reserve(&mut self, bytes: u64) -> Result<(), String> {
        let reserved = self
            .reserved
            .checked_add(bytes)
            .ok_or_else(|| "process quota reservation overflowed".to_owned())?;
        if reserved > self.limit {
            return Err("process quota reservation exceeds limits.live_bytes".to_owned());
        }
        self.reserved = reserved;
        Ok(())
    }

    /// Returns an exact generation charge to the ledger.
    ///
    /// # Errors
    ///
    /// Returns an error if the release exceeds the current reservation.
    pub fn release(&mut self, bytes: u64) -> Result<(), String> {
        self.reserved = self
            .reserved
            .checked_sub(bytes)
            .ok_or_else(|| "process quota release exceeds its reservation".to_owned())?;
        Ok(())
    }

    /// Reports whether an additional reservation fits without overflow.
    #[must_use]
    pub fn fits(&self, bytes: u64) -> bool {
        self.reserved
            .checked_add(bytes)
            .is_some_and(|reserved| reserved <= self.limit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_returns_exact_generation_charges() {
        let mut ledger = ProcessLedger::new(10);
        ledger.reserve(6).unwrap();
        assert!(!ledger.fits(5));
        ledger.release(6).unwrap();
        assert!(ledger.fits(10));
        assert!(ledger.release(1).is_err());
    }
}
