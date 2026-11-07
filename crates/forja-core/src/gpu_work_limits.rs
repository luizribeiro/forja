use std::time::{Duration, Instant};

use crate::BackendError;

/// Host-controlled GPU admission budgets validated against the submission deadline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuWorkLimits {
    dispatch: Duration,
    command_buffer: Duration,
    submission: Duration,
    timeout: Duration,
    backstop_ratio: u32,
}

impl GpuWorkLimits {
    /// Validates positive ordered budgets and a backstop that fits inside the deadline.
    ///
    /// # Errors
    /// Refuses zero, inconsistent or unrepresentable budgets.
    pub fn new(
        dispatch: Duration,
        command_buffer: Duration,
        submission: Duration,
        timeout: Duration,
        backstop_ratio: u32,
    ) -> Result<Self, BackendError> {
        if dispatch.is_zero()
            || dispatch > command_buffer
            || command_buffer > submission
            || submission >= timeout
            || backstop_ratio == 0
            || command_buffer
                .checked_mul(backstop_ratio)
                .is_none_or(|bound| bound >= timeout)
            || Instant::now().checked_add(timeout).is_none()
        {
            return Err(BackendError::InvalidInput);
        }
        Ok(Self {
            dispatch,
            command_buffer,
            submission,
            timeout,
            backstop_ratio,
        })
    }

    /// Resolves two-second command budgets and an eight-second submission budget,
    /// reducing them proportionally when the host selects a shorter timeout.
    ///
    /// # Errors
    /// Refuses a timeout too short to represent positive budgets or too long for a deadline.
    pub fn for_timeout(timeout: Duration) -> Result<Self, BackendError> {
        let submission = (timeout / 5)
            .checked_mul(4)
            .ok_or(BackendError::InvalidInput)?
            .min(Duration::from_secs(8));
        let command = (submission / 4).min(Duration::from_secs(2));
        Self::new(command, command, submission, timeout, 4)
    }

    /// Returns the predicted time allowance for one dispatch.
    #[must_use]
    pub const fn dispatch(self) -> Duration {
        self.dispatch
    }
    /// Returns the predicted time allowance for one command buffer.
    #[must_use]
    pub const fn command_buffer(self) -> Duration {
        self.command_buffer
    }
    /// Returns the summed predicted time allowance for one submission.
    #[must_use]
    pub const fn submission(self) -> Duration {
        self.submission
    }
    /// Returns the wall-clock completion deadline measured from commit.
    #[must_use]
    pub const fn timeout(self) -> Duration {
        self.timeout
    }
    /// Returns the maximum measured/predicted ratio before refusing a composition.
    #[must_use]
    pub const fn backstop_ratio(self) -> u32 {
        self.backstop_ratio
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_budgets_fit_long_and_short_deadlines() {
        for timeout in [
            Duration::from_micros(1),
            Duration::from_secs(1),
            Duration::from_secs(10),
            Duration::from_secs(60),
        ] {
            let limits = GpuWorkLimits::for_timeout(timeout).unwrap();
            assert!(limits.dispatch() <= limits.command_buffer());
            assert!(limits.command_buffer() <= limits.submission());
            assert!(limits.submission() < timeout);
            assert!(limits.command_buffer() * limits.backstop_ratio() < timeout);
        }
        let limits = GpuWorkLimits::for_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(limits.dispatch(), Duration::from_secs(2));
        assert_eq!(limits.submission(), Duration::from_secs(8));
    }

    #[test]
    fn explicit_budgets_refuse_bad_order_overflow_and_deadline_conflicts() {
        let second = Duration::from_secs(1);
        for (dispatch, buffer, submission, timeout, ratio) in [
            (Duration::ZERO, second, second, second * 10, 4),
            (second * 2, second, second * 8, second * 10, 4),
            (second, second * 2, second, second * 10, 4),
            (second, second, second * 10, second * 10, 4),
            (second, second * 3, second * 8, second * 10, 4),
            (second, second, second * 8, second * 10, 0),
            (second, Duration::MAX, Duration::MAX, Duration::MAX, 4),
        ] {
            assert_eq!(
                GpuWorkLimits::new(dispatch, buffer, submission, timeout, ratio),
                Err(BackendError::InvalidInput)
            );
        }
        assert!(GpuWorkLimits::for_timeout(Duration::from_nanos(1)).is_err());
        assert!(GpuWorkLimits::for_timeout(Duration::MAX).is_err());
    }
}
