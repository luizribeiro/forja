use super::*;
use crate::Host;
use forja_core::{StateLayout, StatePlane, StateSequence, Submission};

pub(super) struct Fixture<B: Backend> {
    pub(super) host: Host<B>,
    pub(super) storage: Arc<StateStorage<B>>,
    pub(super) input: Tensor,
    pub(super) weights: Tensor,
    pub(super) delta: Vec<Tensor>,
    pub(super) outputs: Vec<[Tensor; 2]>,
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
        let tensor = |shape: &[u32], value: f32| {
            let tensor = host.backend.alloc(DType::F32, shape).unwrap();
            host.backend
                .write(
                    &tensor,
                    &value
                        .to_le_bytes()
                        .repeat(usize::try_from(tensor.layout().element_count()).unwrap()),
                )
                .unwrap();
            tensor
        };
        Self {
            storage,
            input: tensor(&[1, 7], 2.),
            weights: tensor(&[4, 7], 1.),
            delta: [
                (&[1, 7][..], 0.125),
                (&[1, 7], 0.125),
                (&[1, 33], 2.),
                (&[1], 1.),
                (&[1], 0.5),
            ]
            .iter()
            .map(|(shape, value)| tensor(shape, *value))
            .collect(),
            outputs: (0..2)
                .map(|_| [tensor(&[1, 7], 0.), tensor(&[1, 33], 0.)])
                .collect(),
            host,
        }
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

    pub(super) fn update(&self, commands: &mut StateCommands<B>, row: usize) {
        commands
            .causal_convolution(0, &self.input, &self.weights, &self.outputs[row][0])
            .unwrap();
        commands
            .delta_rule(
                1,
                self.delta.iter().collect::<Vec<_>>().try_into().unwrap(),
                &self.outputs[row][1],
            )
            .unwrap();
    }

    pub(super) fn corrupt_word(&self, tensor: &Tensor, index: usize) {
        let mut bytes = self.host.backend.read(tensor).unwrap();
        let start = index.checked_mul(size_of::<u32>()).unwrap();
        let end = start.checked_add(size_of::<u32>()).unwrap();
        let word = bytes.get_mut(start..end).unwrap();
        let generation = u32::from_le_bytes((&*word).try_into().unwrap()) ^ 1;
        word.copy_from_slice(&generation.to_le_bytes());
        self.host.backend.write(tensor, &bytes).unwrap();
    }

    pub(super) fn read_outputs(&self) -> Vec<Vec<Vec<u8>>> {
        self.outputs
            .iter()
            .map(|row| {
                row.iter()
                    .map(|tensor| self.host.backend.read(tensor).unwrap())
                    .collect()
            })
            .collect()
    }

    pub(super) fn execute_result(
        &self,
        mut commands: StateCommands<B>,
    ) -> Result<(), forja_core::BackendError> {
        self.host
            .backend
            .submit_with_timeout(
                std::mem::take(&mut commands.commands),
                commands.retained_resources(),
                false,
                self.host.limits.submission_timeout,
            )
            .unwrap()
            .wait_timeout(self.host.limits.submission_timeout)
    }
    pub(super) fn execute(&self, commands: StateCommands<B>) {
        self.execute_result(commands).unwrap();
    }
}
