use super::*;
use crate::Host;
use forja_core::{StateLayout, StatePlane, StateSequence};

pub(super) struct Fixture<B: Backend> {
    pub(super) host: Host<B>,
    pub(super) storage: Arc<StateStorage<B>>,
}

impl<B: Backend + 'static> Fixture<B> {
    pub(super) fn new(backend: B) -> Self {
        let host = Host::new(backend, Limits::new(1_048_576, 8, 4096, 128, 1_048_576));
        let storage = host
            .create_state_storage(
                StateLayout::new(vec![
                    StatePlane::new(DType::F32, vec![3, 7]).unwrap(),
                    StatePlane::new(DType::F32, vec![1, 7, 33]).unwrap(),
                ])
                .unwrap(),
                2,
            )
            .unwrap();
        Self { host, storage }
    }

    pub(super) fn reserve(&self) -> StateSequence {
        let sequence = self.storage.pool.reserve().unwrap();
        for entry in sequence.allocation_table().unwrap().entries() {
            for plane in 0..self.storage.planes.len() {
                let view = self.storage.version_view(plane, entry.page()).unwrap();
                self.host
                    .backend
                    .write(
                        &view,
                        &vec![0; usize::try_from(view.layout().element_count() * 4).unwrap()],
                    )
                    .unwrap();
            }
        }
        sequence
    }

    pub(super) fn commands(
        &self,
        sequence: &mut StateSequence,
        serial: u64,
        count: u32,
    ) -> StateCommands<B> {
        let tx = sequence.begin_chain(serial, count).unwrap();
        let slot = sequence.bind(&tx).unwrap();
        let mut commands = self
            .storage
            .prepare_record_metadata(StateAccess::from_slot(&slot).unwrap())
            .unwrap();
        commands.initialize_slot(&slot).unwrap();
        commands
    }
}
