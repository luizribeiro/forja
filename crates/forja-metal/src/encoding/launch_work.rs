use super::{BackendError, MTLSize};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct LaunchWork {
    pub(super) attention: Option<super::AttentionSchedule>,
    pub(super) groups: u64,
    pub(super) threads: u64,
    pub(super) launches: u64,
}

thread_local! { static CURRENT: Cell<Option<LaunchWork>> = const { Cell::new(None) }; }

pub(super) struct Meter(Option<LaunchWork>);
impl Meter {
    pub(super) fn enter() -> Self {
        Self(CURRENT.replace(Some(LaunchWork::default())))
    }
    pub(super) fn finish(self) -> LaunchWork {
        let work = CURRENT.get().unwrap_or_default();
        drop(self);
        work
    }
}
impl Drop for Meter {
    fn drop(&mut self) {
        CURRENT.set(self.0);
    }
}

pub(super) fn record(groups: MTLSize, threads: MTLSize) -> Result<(), BackendError> {
    let product = |size: MTLSize| {
        u64::try_from(size.width)
            .ok()?
            .checked_mul(u64::try_from(size.height).ok()?)?
            .checked_mul(u64::try_from(size.depth).ok()?)
    };
    CURRENT.with(|current| {
        let Some(mut work) = current.get() else {
            return Ok(());
        };
        let groups = product(groups).ok_or(BackendError::InvalidInput)?;
        let threads = product(threads)
            .and_then(|n| n.checked_mul(groups))
            .ok_or(BackendError::InvalidInput)?;
        work.groups = work
            .groups
            .checked_add(groups)
            .ok_or(BackendError::InvalidInput)?;
        work.threads = work
            .threads
            .checked_add(threads)
            .ok_or(BackendError::InvalidInput)?;
        work.launches = work
            .launches
            .checked_add(1)
            .ok_or(BackendError::InvalidInput)?;
        current.set(Some(work));
        Ok(())
    })
}

pub(super) fn attention(schedule: super::AttentionSchedule) {
    CURRENT.with(|current| {
        if let Some(mut work) = current.get() {
            work.attention = Some(schedule);
            current.set(Some(work));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn launch_meter_counts_all_dimensions_and_restores_nested_scopes() {
        let outer = Meter::enter();
        record(
            MTLSize {
                width: 7,
                height: 33,
                depth: 2,
            },
            MTLSize {
                width: 32,
                height: 4,
                depth: 1,
            },
        )
        .unwrap();
        let inner = Meter::enter();
        assert!(
            record(
                MTLSize {
                    width: usize::MAX,
                    height: 33,
                    depth: 1
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1
                }
            )
            .is_err()
        );
        assert_eq!(inner.finish(), LaunchWork::default());
        assert_eq!(
            outer.finish(),
            LaunchWork {
                attention: None,
                groups: 462,
                threads: 59136,
                launches: 1
            }
        );
    }
}
