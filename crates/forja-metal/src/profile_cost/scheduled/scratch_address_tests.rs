use super::addressing;
use forja_core::{
    BackendDispatchData, CommandList, DType, DeviceCapability, Dispatch, KernelCost, Op, ViewOp,
};

fn assert_previous_scratch_bound(dispatch: &Dispatch) {
    for bytes in [1_u64, 7, 33, 4097] {
        let cost = KernelCost {
            read_bytes: bytes,
            scratch_read_bytes: bytes,
            ..KernelCost::default()
        };
        let previous = bytes.div_ceil(2) * 32;
        assert!(addressing(dispatch, cost).unwrap().scalar_ops >= previous);
    }
}

#[test]
fn vector_attention_scratch_retains_the_previous_address_bound() {
    let cpu = forja_cpu::CpuBackend::new();
    let query = cpu.alloc(DType::F32, &[32, 1, 128]).unwrap();
    let key = cpu.alloc(DType::F32, &[4, 4097, 128]).unwrap();
    let value = cpu.alloc(DType::F32, &[4, 4097, 128]).unwrap();
    let output = cpu.alloc(DType::F32, &[32, 1, 128]).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::Sdpa {
                scale: 1.0,
                causal: true,
                q_start: 4096,
            },
            &[&query, &key, &value],
            &output,
        )
        .unwrap();
    let variant = crate::variants::resolve(
        commands.last_dispatch().unwrap(),
        "sdpa.vector-gqa-two-pass-32",
        &[DeviceCapability::Metal4],
    )
    .unwrap();
    commands.set_last_backend_data(BackendDispatchData::new(variant));
    assert_previous_scratch_bound(commands.last_dispatch().unwrap());
}

#[test]
fn strided_sampling_scratch_retains_the_previous_address_bound() {
    let cpu = forja_cpu::CpuBackend::new();
    let backing = cpu.alloc(DType::F32, &[4097, 7]).unwrap();
    let logits = cpu.view(&backing, ViewOp::Permute(vec![1, 0])).unwrap();
    let controls = cpu.alloc(DType::U32, &[5]).unwrap();
    let mask = cpu.alloc(DType::U32, &[129]).unwrap();
    let output = cpu.alloc(DType::U32, &[7]).unwrap();
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::SampleMasked {
                position: 7,
                chunk_width: 2048,
                fast_top_k: 8,
                fast_max_chunks: 32,
                rejection_rounds: 4,
            },
            &[&logits, &controls, &mask],
            &output,
        )
        .unwrap();
    assert_previous_scratch_bound(commands.last_dispatch().unwrap());
}
