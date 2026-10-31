use crate::compute;
use forja_core::BackendError;
use std::fmt;

#[derive(Debug, PartialEq)]
pub(crate) enum RecoveryError<P, C = P> {
    Primary(P),
    Cleanup(C),
    Both { primary: P, cleanup: C },
}

impl<P, C> From<P> for RecoveryError<P, C> {
    fn from(primary: P) -> Self {
        Self::Primary(primary)
    }
}

impl<P: fmt::Debug, C: fmt::Debug> fmt::Display for RecoveryError<P, C> {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primary(error) => write!(out, "{error:?}"),
            Self::Cleanup(error) => write!(out, "cleanup failed: {error:?}"),
            Self::Both { primary, cleanup } => {
                write!(out, "{primary:?}; cleanup failed: {cleanup:?}")
            }
        }
    }
}

pub(crate) fn finish<T, P, C>(
    primary: Result<T, P>,
    cleanup: Result<(), C>,
) -> Result<T, RecoveryError<P, C>> {
    match (primary, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(primary), Ok(())) => Err(RecoveryError::Primary(primary)),
        (Ok(_), Err(cleanup)) => Err(RecoveryError::Cleanup(cleanup)),
        (Err(primary), Err(cleanup)) => Err(RecoveryError::Both { primary, cleanup }),
    }
}

impl RecoveryError<compute::Error> {
    pub(crate) fn into_compute(self) -> compute::Error {
        match self {
            Self::Primary(error) | Self::Cleanup(error) => error,
            Self::Both {
                mut primary,
                cleanup,
            } => {
                let context = format!("; cleanup failed: {cleanup:?}");
                match &mut primary {
                    compute::Error::Layout(message)
                    | compute::Error::OpSignature(message)
                    | compute::Error::Quota(message)
                    | compute::Error::BackendExecution(message)
                    | compute::Error::InvalidHandle(message) => message.push_str(&context),
                }
                primary
            }
        }
    }
}

impl RecoveryError<BackendError, compute::Error> {
    pub(crate) fn into_backend(self) -> BackendError {
        match self {
            Self::Primary(error) => error,
            error @ Self::Both { .. } => {
                eprintln!("submission recovery: {error}");
                let Self::Both { primary, .. } = error else {
                    return BackendError::ExecutionFailed;
                };
                primary
            }
            error @ Self::Cleanup(_) => {
                eprintln!("submission recovery: {error}");
                BackendError::ExecutionFailed
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn double_failure_preserves_primary_kind_and_cleanup_context() {
        let failure = finish::<(), _, _>(
            Err(compute::Error::Quota("original".into())),
            Err(compute::Error::InvalidHandle("restore".into())),
        )
        .unwrap_err();
        let compute::Error::Quota(message) = failure.into_compute() else {
            panic!("lost primary kind")
        };
        assert!(message.contains("original") && message.contains("restore"));
        let failure = finish::<(), _, _>(
            Err(BackendError::Timeout),
            Err(compute::Error::InvalidHandle("journal".into())),
        )
        .unwrap_err();
        assert!(failure.to_string().contains("journal"));
        assert_eq!(failure.into_backend(), BackendError::Timeout);
        assert!(matches!(
            finish::<(), (), _>(Ok(()), Err("cleanup")),
            Err(RecoveryError::Cleanup("cleanup"))
        ));
    }
}
