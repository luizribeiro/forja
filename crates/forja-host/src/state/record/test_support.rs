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
        Self::with_spares(backend, 0)
    }

    pub(super) fn with_spares(backend: B, spares: u32) -> Self {
        let host = Host::new(backend, Limits::new(1_048_576, 8, 4096, 128, 1_048_576));
        let storage = host
            .create_state_storage_with_spares(
                StateLayout::new(vec![
                    StatePlane::new(DType::F32, vec![3, 7]).unwrap(),
                    StatePlane::new(DType::F32, vec![1, 7, 33]).unwrap(),
                ])
                .unwrap(),
                2,
                spares,
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
            outputs: (0..2 + spares)
                .map(|_| [tensor(&[1, 7], 0.), tensor(&[1, 33], 0.)])
                .collect(),
            host,
        }
    }

    pub(super) fn reserve(&self) -> StateSequence {
        self.reserve_with_spares(false)
    }

    pub(super) fn reserve_with_spares(&self, spares: bool) -> StateSequence {
        let mut sequence = self.storage.pool.reserve().unwrap();
        if spares {
            sequence.reserve_spares().unwrap();
        }
        self.storage
            .clear_records(sequence.allocation_table().unwrap().entries())
            .unwrap();
        sequence
    }

    pub(super) fn commands(
        &self,
        sequence: &mut StateSequence,
        serial: u64,
        count: u32,
    ) -> StateCommands<B> {
        self.begin_commands(sequence, serial, count).1
    }

    pub(super) fn begin_commands(
        &self,
        sequence: &mut StateSequence,
        serial: u64,
        count: u32,
    ) -> (StateTransaction, StateCommands<B>) {
        let tx = sequence.begin_chain(serial, count).unwrap();
        let slot = sequence.bind(&tx).unwrap();
        let mut commands = self
            .storage
            .prepare_record_metadata(StateAccess::from_slot(&slot).unwrap())
            .unwrap();
        commands.initialize_slot(&slot).unwrap();
        (tx, commands)
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
