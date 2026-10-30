use super::*;
use crate::{Host, LanePosition, Limits};
use forja_core::{DType, StateLayout, StatePlane};

fn queued_cancellation<B: Backend + Send + Sync + 'static>(backend: B) {
    let host = Host::new(backend, Limits::new(131_072, 8, 4096, 128, 4096));
    let storage = host
        .create_state_storage(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
            2,
        )
        .unwrap();
    let lanes = [storage.reserve().unwrap(), storage.reserve().unwrap()];
    let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
    let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
    let outputs = [
        host.backend.alloc(DType::F32, &[1, 7]).unwrap(),
        host.backend.alloc(DType::F32, &[1, 7]).unwrap(),
    ];
    host.backend
        .write(&input, &2_f32.to_le_bytes().repeat(7))
        .unwrap();
    host.backend
        .write(&weights, &1_f32.to_le_bytes().repeat(28))
        .unwrap();
    let position = |tokens| LanePosition {
        tokens,
        sampler: tokens,
        grammar: tokens,
        parser: tokens,
        reserved_tokens: 128,
    };
    let batch = |serial, tokens| {
        StateBatch::new(
            lanes
                .iter()
                .zip(&outputs)
                .map(|(lane, output)| {
                    let mut recording = storage
                        .record(lane, lane.begin_at(serial, position(tokens)).unwrap())
                        .unwrap();
                    recording
                        .causal_convolution(0, &input, &weights, output)
                        .unwrap();
                    recording
                })
                .collect(),
        )
        .unwrap()
    };
    let first = batch(1, 1).submit().unwrap();
    let second = batch(2, 2).submit().unwrap();
    assert_eq!(lanes[0].position(), LanePosition::default());
    first.cancel().unwrap();
    assert!(lanes[0].begin(3).is_err());
    assert!(second.finish().is_err());
    batch(3, 1).submit_and_finish().unwrap();
    for (lane, output) in lanes.iter().zip(&outputs) {
        assert_eq!(lane.position(), position(1));
        assert_eq!(
            host.backend.read(output).unwrap(),
            2_f32.to_le_bytes().repeat(7)
        );
    }
    let first = batch(4, 2).submit().unwrap();
    let second = batch(5, 3).submit().unwrap();
    drop(first);
    assert!(second.finish().is_err());
    for lane in &lanes {
        assert_eq!(lane.position(), position(1));
    }
    batch(6, 2).submit_and_finish().unwrap();
    let first = batch(7, 3).submit().unwrap();
    let second = batch(8, 4);
    std::thread::scope(|scope| {
        let (sender, receiver) = std::sync::mpsc::channel();
        let worker = scope.spawn(move || {
            sender
                .send(second.submit().and_then(StateBatchSubmission::finish))
                .unwrap();
        });
        assert!(matches!(
            receiver.recv_timeout(std::time::Duration::from_millis(100)),
            Err(std::sync::mpsc::RecvTimeoutError::Timeout)
        ));
        first.finish().unwrap();
        receiver
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap()
            .unwrap();
        worker.join().unwrap();
    });
    for lane in &lanes {
        assert_eq!(lane.position(), position(4));
    }
}

#[test]
fn depth_two_state_batches_cancel_and_drain_before_lane_reuse() {
    queued_cancellation(forja_cpu::CpuBackend::new());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_depth_two_state_batches_cancel_and_drain_before_lane_reuse() {
    queued_cancellation(forja_metal::MetalBackend::new().unwrap());
}
