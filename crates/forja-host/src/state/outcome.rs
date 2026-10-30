use forja_core::BackendError;
use std::{
    sync::{Condvar, Mutex},
    time::{Duration, Instant},
};

#[derive(Default)]
pub(super) struct StateOutcome {
    result: Mutex<Option<bool>>,
    ready: Condvar,
}

impl StateOutcome {
    pub(super) fn finish(&self, success: bool) {
        let mut result = match self.result.lock() {
            Ok(result) => result,
            Err(poisoned) => poisoned.into_inner(),
        };
        if result.is_none() {
            *result = Some(success);
        }
        self.ready.notify_all();
    }

    pub(super) fn wait(&self, timeout: Duration) -> Result<bool, BackendError> {
        let started = Instant::now();
        let mut result = self
            .result
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        loop {
            if let Some(success) = *result {
                return Ok(success);
            }
            let remaining = timeout.saturating_sub(started.elapsed());
            if remaining.is_zero() {
                return Err(BackendError::Timeout);
            }
            result = self
                .ready
                .wait_timeout(result, remaining)
                .map_err(|_| BackendError::ExecutionFailed)?
                .0;
        }
    }
}
