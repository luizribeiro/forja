use super::*;
use crate::{Host, Limits};
use forja_core::{DType, StateLayout, StatePlane, Submission};
use std::time::Duration;

fn poisoned_padding<B: Backend + Send + Sync + 'static>(backend: B) {
    let host = Host::new(backend, Limits::new(131_072, 8, 4096, 128, 4096));
    let storage = host
        .create_state_storage(
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
            2,
        )
        .unwrap();
    let mut slots = StateSlots::new(Arc::clone(&storage), 2).unwrap();
    let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
    let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
    let output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
    let previous = storage
        .version_view(0, slots.inactive_access.source())
        .unwrap();
    let next = storage
        .version_view(0, slots.inactive_access.destination())
        .unwrap();
    host.backend
        .write(&input, &2_f32.to_le_bytes().repeat(7))
        .unwrap();
    host.backend
        .write(&weights, &1_f32.to_le_bytes().repeat(28))
        .unwrap();
    host.backend
        .write(&previous, &9_f32.to_le_bytes().repeat(21))
        .unwrap();
    host.backend
        .write(&next, &13_f32.to_le_bytes().repeat(21))
        .unwrap();
    for corrupt in [false, true] {
        let mut commands = slots.inactive().unwrap();
        commands
            .causal_convolution(0, &input, &weights, &output)
            .unwrap();
        host.backend
            .write(&output, &9_f32.to_le_bytes().repeat(7))
            .unwrap();
        if corrupt {
            host.backend
                .write(&commands.metadata[2], &1_u32.to_le_bytes())
                .unwrap();
        }
        let submission = host
            .backend
            .submit_retained(
                std::mem::take(&mut commands.commands),
                commands.retained_resources(),
            )
            .unwrap();
        drop(commands);
        assert_eq!(
            submission.wait_timeout(Duration::from_secs(10)).is_err(),
            corrupt
        );
        assert_eq!(
            host.backend.read_replay_output(&previous).unwrap(),
            9_f32.to_le_bytes().repeat(21)
        );
        if !corrupt {
            assert_eq!(
                host.backend.read(&output).unwrap(),
                0_f32.to_le_bytes().repeat(7)
            );
            assert_eq!(
                host.backend.read(&next).unwrap(),
                13_f32.to_le_bytes().repeat(21)
            );
        }
    }
    let mut commands = slots.inactive().unwrap();
    commands
        .causal_convolution(0, &input, &weights, &output)
        .unwrap();
    let capture = commands.capture().unwrap();
    let template = std::mem::take(&mut commands.commands);
    drop(commands);
    slots.admit(0).unwrap();
    slots.admit(1).unwrap();
    slots
        .begin(
            0,
            0,
            LanePosition {
                tokens: 1,
                reserved_tokens: 128,
                ..LanePosition::default()
            },
        )
        .unwrap();
    let recording = slots.recording(0, 0).unwrap();
    let bindings = capture.bind(recording).unwrap();
    recording.commands = bindings.apply(template).unwrap();
    slots.take_batch(0).unwrap().submit_and_finish().unwrap();
    assert_eq!(
        host.backend.read(&output).unwrap(),
        2_f32.to_le_bytes().repeat(7)
    );
    drop(capture);
    slots.release(0).unwrap();
    slots.release(1).unwrap();
}

#[test]
fn inactive_commands_touch_no_state_and_hold_no_lane_lease() {
    poisoned_padding(forja_cpu::CpuBackend::new());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_inactive_commands_touch_no_state_and_hold_no_lane_lease() {
    poisoned_padding(forja_metal::MetalBackend::new().unwrap());
}
