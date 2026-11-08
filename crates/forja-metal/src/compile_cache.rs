use std::{
    collections::HashMap,
    hash::Hash,
    sync::{Arc, Mutex, OnceLock},
};

use forja_core::BackendError;

type Slot<V> = Arc<OnceLock<Result<V, BackendError>>>;

pub(super) struct CompileCache<K, V> {
    entries: Mutex<HashMap<K, Slot<V>>>,
}

impl<K, V> Default for CompileCache<K, V> {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }
}

impl<K: Eq + Hash, V: Clone> CompileCache<K, V> {
    pub(super) fn get(&self, key: &K) -> Result<Option<V>, BackendError> {
        let slot = self
            .entries
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .get(key)
            .cloned();
        slot.and_then(|slot| slot.get().cloned()).transpose()
    }

    pub(super) fn contains_key(&self, key: &K) -> Result<bool, BackendError> {
        Ok(self
            .entries
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?
            .contains_key(key))
    }

    pub(super) fn compile(
        &self,
        key: K,
        admit: impl FnOnce() -> Result<(), BackendError>,
        build: impl FnOnce() -> Result<V, BackendError>,
    ) -> Result<V, BackendError> {
        let slot = {
            let mut entries = self
                .entries
                .lock()
                .map_err(|_| BackendError::ExecutionFailed)?;
            match entries.entry(key) {
                std::collections::hash_map::Entry::Occupied(entry) => Arc::clone(entry.get()),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    admit()?;
                    Arc::clone(entry.insert(Arc::new(OnceLock::new())))
                }
            }
        };
        slot.get_or_init(build).clone()
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    #[cfg(test)]
    pub(super) fn insert(&self, key: K, value: V) {
        self.entries
            .lock()
            .unwrap()
            .insert(key, Arc::new(OnceLock::from(Ok(value))));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        mpsc,
    };

    #[test]
    fn compiling_one_key_does_not_lock_out_other_keys() {
        let cache = Arc::new(CompileCache::default());
        let (entered, ready) = mpsc::channel();
        let (release, resume) = mpsc::channel();
        let worker_cache = Arc::clone(&cache);
        let worker = std::thread::spawn(move || {
            worker_cache.compile(
                "blocked",
                || Ok(()),
                || {
                    entered.send(()).unwrap();
                    resume.recv().unwrap();
                    Ok(33_u32)
                },
            )
        });
        ready.recv().unwrap();
        assert_eq!(cache.compile("other", || Ok(()), || Ok(7)), Ok(7));
        release.send(()).unwrap();
        assert_eq!(worker.join().unwrap(), Ok(33));
        assert_eq!(cache.get(&"blocked").unwrap(), Some(33));
    }

    #[test]
    fn failure_and_success_each_reserve_only_once_per_key() {
        let cache = CompileCache::<_, u32>::default();
        let admissions = AtomicUsize::new(0);
        for _ in 0..2 {
            assert_eq!(
                cache.compile(
                    7,
                    || {
                        admissions.fetch_add(1, Ordering::Relaxed);
                        Ok(())
                    },
                    || Err(BackendError::ExecutionFailed)
                ),
                Err(BackendError::ExecutionFailed)
            );
        }
        assert_eq!(admissions.load(Ordering::Relaxed), 1);
        assert_eq!(
            cache.compile(
                33,
                || Err(BackendError::QuotaExceeded),
                || panic!("unadmitted compilation ran")
            ),
            Err(BackendError::QuotaExceeded)
        );
        assert!(!cache.contains_key(&33).unwrap());
    }
}
