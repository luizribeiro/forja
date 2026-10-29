use crate::{Host, Limits};
use forja_core::{Backend, DType, StateLayout, StatePlane};

fn independent_serials<B: Backend + 'static>(backend: B) {
    let host = Host::new(backend, Limits::new(65536, 8, 4096, 128, 4096));
    let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![1, 1]).unwrap()]).unwrap();
    let storage = host.create_state_storage(layout, 2).unwrap();
    let mut a = storage.reserve().unwrap();
    let mut b = storage.reserve().unwrap();
    let x = host.backend.alloc(DType::F32, &[1, 1]).unwrap();
    let weights = host.backend.alloc(DType::F32, &[2, 1]).unwrap();
    let y_a = host.backend.alloc(DType::F32, &[1, 1]).unwrap();
    let y_b = host.backend.alloc(DType::F32, &[1, 1]).unwrap();
    host.backend.write(&x, &2_f32.to_le_bytes()).unwrap();
    host.backend
        .write(
            &weights,
            &[1_f32.to_le_bytes(), 1_f32.to_le_bytes()].concat(),
        )
        .unwrap();
    let first = a.begin(1).unwrap();
    let second = b.begin(1).unwrap();
    let mut first = storage.record(&a, first).unwrap();
    let mut second = storage.record(&b, second).unwrap();
    first.causal_convolution(0, &x, &weights, &y_a).unwrap();
    second.causal_convolution(0, &x, &weights, &y_b).unwrap();
    let first = first.submit(&mut a).unwrap();
    let second = second.submit(&mut b).unwrap();
    second.finish(&mut b).unwrap();
    first.finish(&mut a).unwrap();
    assert_eq!(host.backend.read(&y_a).unwrap(), 2_f32.to_le_bytes());
    assert_eq!(host.backend.read(&y_b).unwrap(), 2_f32.to_le_bytes());
    assert_eq!(a.begin(2).unwrap().input_version(), 1);
    assert_eq!(b.begin(2).unwrap().input_version(), 1);
}

#[test]
fn same_pool_sequences_can_both_submit_serial_one() {
    independent_serials(forja_cpu::CpuBackend::new());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_same_pool_sequences_can_both_submit_serial_one() {
    independent_serials(forja_metal::MetalBackend::new().unwrap());
}
