use super::*;
use crate::Host;
use forja_core::{StateLayout, StatePlane};

#[test]
fn single_output_access_preserves_dispatch_and_publication() {
    let host = Host::new(
        forja_cpu::CpuBackend::new(),
        Limits::new(65536, 8, 4096, 128, 4096),
    );
    let storage = host
        .create_state_storage(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
            1,
        )
        .unwrap();
    let sequence = storage.reserve().unwrap();
    let mut recording = storage
        .record(&sequence, sequence.begin(1).unwrap())
        .unwrap();
    assert_eq!(recording.access.chain_len(), 1);
    assert_eq!(recording.metadata[0].layout().shape(), [2, 2]);
    let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
    let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
    let output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
    host.backend
        .write(&input, &2_f32.to_le_bytes().repeat(7))
        .unwrap();
    host.backend
        .write(&weights, &1_f32.to_le_bytes().repeat(28))
        .unwrap();
    recording
        .causal_convolution(0, &input, &weights, &output)
        .unwrap();
    let Op::State(operation) = recording.commands.dispatches()[0].op() else {
        unreachable!()
    };
    assert_eq!(operation.access(), recording.access.step(0).unwrap());
    recording.submit().unwrap().finish().unwrap();
    assert_eq!(
        host.backend.read(&output).unwrap(),
        2_f32.to_le_bytes().repeat(7)
    );
}
