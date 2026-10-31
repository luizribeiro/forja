use super::*;
use crate::{Host, Limits, StateAction, StateStorage};
use forja_core::{
    DType, HybridPrefixNamespace, PagedLayout, PagedPlane, StateLayout, StatePlane, ViewOp,
};
use forja_testing::{DeterministicValues, TensorSpec, generated_tensor_bytes};
use std::num::NonZeroU32;

struct BlockInputs {
    delta: [Tensor; 5],
    output: Tensor,
    conv: Tensor,
    weights: Tensor,
    conv_output: Tensor,
}
impl BlockInputs {
    fn new(backend: &impl Backend, seed: u64) -> Self {
        let mut values = DeterministicValues::new(seed);
        let mut tensor = |dtype, shape: &[u32]| {
            let tensor = backend.alloc(dtype, shape).unwrap();
            let bytes =
                generated_tensor_bytes(&TensorSpec::contiguous(dtype, shape), &mut values).unwrap();
            backend.write(&tensor, &bytes).unwrap();
            tensor
        };
        let query = tensor(DType::F32, &[7, 1, 64]);
        let query = backend
            .view(&query, ViewOp::Permute(vec![2, 1, 0]))
            .unwrap();
        let delta = [
            query,
            tensor(DType::F32, &[64, 1, 7]),
            tensor(DType::F32, &[64, 1, 33]),
            tensor(DType::F32, &[64, 1]),
            tensor(DType::F32, &[64, 1]),
        ];
        backend
            .write(&delta[3], &(-0.03125_f32).to_le_bytes().repeat(64))
            .unwrap();
        backend
            .write(&delta[4], &0.03125_f32.to_le_bytes().repeat(64))
            .unwrap();
        Self {
            delta,
            output: backend.alloc(DType::F32, &[64, 1, 33]).unwrap(),
            conv: tensor(DType::BF16, &[64, 7]),
            weights: tensor(DType::BF16, &[4, 7]),
            conv_output: backend.alloc(DType::BF16, &[64, 7]).unwrap(),
        }
    }
    fn run<B: Backend + 'static>(
        &self,
        storage: &Arc<StateStorage<B>>,
        sequence: &StateSequenceHandle<B>,
        prompt: &Arc<CanonicalPrompt>,
        serial: u64,
        through: u64,
    ) -> [Vec<u8>; 2] {
        let transaction = sequence
            .begin_prefill(serial, prompt.clone(), through, 257, StateAction::Step)
            .unwrap();
        let mut recording = storage.record(sequence, transaction).unwrap();
        recording
            .delta_rule_block(
                0,
                self.delta.each_ref(),
                &self.output,
                Some("delta-rule-block.wy64-fixed"),
            )
            .unwrap();
        recording
            .causal_convolution(1, &self.conv, &self.weights, &self.conv_output)
            .unwrap();
        recording.submit().unwrap().finish().unwrap();
        [
            storage.backend.read(&self.output).unwrap(),
            storage.backend.read(&self.conv_output).unwrap(),
        ]
    }
}

fn cold_resume<B: Backend + 'static>(backend: B) {
    let host = Host::new(
        backend,
        Limits::new(4_194_304, 8, 1_048_576, 128, 1_000_000_000),
    );
    let layout = StateLayout::new(vec![
        StatePlane::new(DType::F32, vec![1, 7, 33]).unwrap(),
        StatePlane::new(DType::BF16, vec![3, 7]).unwrap(),
    ])
    .unwrap();
    let pages = PagedLayout::new(
        64,
        vec![PagedPlane {
            dtype: DType::F16,
            elements_per_token: 33,
            alignment: 2,
            label: "key".into(),
        }],
    )
    .unwrap();
    let namespace = HybridPrefixNamespace::new(
        [3; 32],
        &layout,
        &pages,
        NonZeroU32::new(64).unwrap(),
        &["fixed-tree"],
    )
    .unwrap();
    let prompt = host
        .canonical_prompt(namespace, (0..129).collect())
        .unwrap();
    let storage = host.create_state_storage(layout, 3).unwrap();
    let cold = storage.reserve().unwrap();
    let resumed = storage.reserve().unwrap();
    let branch = storage.reserve().unwrap();
    BlockInputs::new(host.backend.as_ref(), 7).run(&storage, &cold, &prompt, 1, 64);
    let allowance = SnapshotAllowance::new(131_072, None).unwrap();
    let image = cold.capture_snapshot(allowance).unwrap();
    let before = image
        .planes
        .iter()
        .map(|plane| host.backend.read(plane).unwrap())
        .collect::<Vec<_>>();
    resumed
        .attach_snapshot(image.clone(), namespace, 257)
        .unwrap();
    branch
        .attach_snapshot(image.clone(), namespace, 257)
        .unwrap();
    let suffix = BlockInputs::new(host.backend.as_ref(), 33);
    let expected = suffix.run(&storage, &cold, &prompt, 2, 128);
    let actual = suffix.run(&storage, &resumed, &prompt, 1, 128);
    assert_eq!(expected, actual);
    let expected = cold.capture_snapshot(allowance).unwrap();
    let actual = resumed.capture_snapshot(allowance).unwrap();
    for (expected, actual) in expected.planes.iter().zip(&actual.planes) {
        assert_eq!(
            host.backend.read(expected).unwrap(),
            host.backend.read(actual).unwrap()
        );
    }
    BlockInputs::new(host.backend.as_ref(), 4097).run(&storage, &branch, &prompt, 1, 128);
    let different = branch.capture_snapshot(allowance).unwrap();
    assert_ne!(
        host.backend.read(&expected.planes[0]).unwrap(),
        host.backend.read(&different.planes[0]).unwrap()
    );
    for (plane, bytes) in image.planes.iter().zip(before) {
        assert_eq!(host.backend.read(plane).unwrap(), bytes);
    }
    assert_eq!(resumed.canonical_token_count().unwrap(), Some(128));
    assert_eq!(branch.canonical_token_count().unwrap(), Some(128));
}

#[test]
fn cpu_cold_and_resumed_hybrid_planes_and_outputs_match_bitwise() {
    cold_resume(forja_cpu::CpuBackend::new());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_cold_and_resumed_hybrid_planes_and_outputs_match_bitwise() {
    cold_resume(forja_metal::MetalBackend::new().unwrap());
}
