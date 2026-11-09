use forja_core::{Backend, CommandList, DType, Op, Submission, Tensor};
use forja_cpu::CpuBackend;
use forja_host::SamplingParams;

pub(super) struct Target {
    backend: CpuBackend,
    logits: Tensor,
    parameters: Tensor,
    output: Tensor,
    pub(super) vocab: u32,
}

impl Target {
    pub(super) fn new(vocab: u32, sampling: SamplingParams) -> Self {
        assert!((2..=u32::from(u16::MAX)).contains(&vocab));
        let backend = CpuBackend::new();
        let logits = backend.alloc(DType::F32, &[vocab]).unwrap();
        let parameters = backend.alloc(DType::U32, &[5]).unwrap();
        let output = backend.alloc(DType::U32, &[]).unwrap();
        let seed = sampling.seed.to_le_bytes();
        backend
            .write(
                &parameters,
                &[
                    sampling.temperature.to_bits(),
                    sampling.top_k,
                    sampling.top_p.to_bits(),
                    u32::from_le_bytes(seed[..4].try_into().unwrap()),
                    u32::from_le_bytes(seed[4..].try_into().unwrap()),
                ]
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
            )
            .unwrap();
        Self {
            backend,
            logits,
            parameters,
            output,
            vocab,
        }
    }

    pub(super) fn sample(&self, prefix: &[u32]) -> u32 {
        let position = u32::try_from(prefix.len().checked_sub(1).unwrap()).unwrap();
        let state = prefix.iter().fold(0_u64, |state, token| {
            (state * 31 + u64::from(*token)) % u64::from(self.vocab)
        });
        let logits: Vec<u8> = (0..self.vocab)
            .flat_map(|index| {
                let distance =
                    (u64::from(index) + u64::from(self.vocab) - state) % u64::from(self.vocab);
                (-f32::from(u16::try_from(distance).unwrap()) / 16.0).to_le_bytes()
            })
            .collect();
        self.backend.write(&self.logits, &logits).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::Sample {
                    position,
                    chunk_width: 2048,
                    fast_top_k: 20,
                    fast_max_chunks: 102,
                    rejection_rounds: 2,
                },
                &[&self.logits, &self.parameters],
                &self.output,
            )
            .unwrap();
        self.backend.submit(commands).unwrap().wait().unwrap();
        u32::from_le_bytes(self.backend.read(&self.output).unwrap().try_into().unwrap())
    }
}

#[test]
fn target_uses_prefix_logits_and_counter_seeded_cpu_sampling() {
    let greedy = Target::new(7, SamplingParams::default());
    assert_eq!(greedy.sample(&[1]), 1);
    assert_eq!(greedy.sample(&[1, 7]), 3);
    let sample = |seed, token| {
        let target = Target::new(
            7,
            SamplingParams {
                temperature: 0.7,
                top_k: 3,
                top_p: 0.9,
                seed,
            },
        );
        (1..=33)
            .map(|len| target.sample(&vec![token; len]))
            .collect::<Vec<_>>()
    };
    assert_eq!(sample(7, 1), sample(7, 1));
    assert_ne!(sample(7, 1), sample(33, 1));
    assert_ne!(sample(7, 1), sample(7, 2));
}
