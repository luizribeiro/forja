use super::*;
use crate::{Host, Limits};
use forja_core::{DType, StateLayout, StatePlane, Submission};
use std::time::Duration;

fn forge_activity<B: Backend>(backend: &B, commands: &StateCommands<B>) {
    let table = [
        commands.access.source(),
        u32::MAX,
        commands.access.destination(),
        u32::MAX,
    ];
    backend
        .write(
            &commands.metadata[0],
            &table
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    backend
        .write(&commands.metadata[2], &1_u32.to_le_bytes())
        .unwrap();
}

fn write_constant<B: Backend>(backend: &B, tensor: &forja_core::Tensor, value: f32) {
    backend
        .write(
            tensor,
            &value
                .to_le_bytes()
                .repeat(usize::try_from(tensor.layout().element_count()).unwrap()),
        )
        .unwrap();
}

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
    write_constant(host.backend.as_ref(), &input, 2.0);
    write_constant(host.backend.as_ref(), &weights, 1.0);
    write_constant(host.backend.as_ref(), &previous, 9.0);
    write_constant(host.backend.as_ref(), &next, 13.0);
    let mut completed = Vec::new();
    for corrupt in [false, true, false] {
        let mut commands = slots.inactive().unwrap();
        commands
            .causal_convolution(0, &input, &weights, &output)
            .unwrap();
        write_constant(host.backend.as_ref(), &output, 9.0);
        if corrupt {
            forge_activity(host.backend.as_ref(), &commands);
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
        }
        assert_eq!(
            host.backend.read(&next).unwrap(),
            13_f32.to_le_bytes().repeat(21)
        );
        completed.push(submission);
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
    let active_output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
    let mut bindings = capture.bind(recording).unwrap();
    bindings
        .buffer(output.buffer(), active_output.buffer())
        .unwrap();
    recording.commands = bindings.apply(template).unwrap();
    slots.take_batch(0).unwrap().submit_and_finish().unwrap();
    assert_eq!(
        host.backend.read(&active_output).unwrap(),
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

#[test]
fn recording_quota_refusal_keeps_the_retained_predecessor() {
    quota_refusal_keeps_predecessor(false);
}

#[test]
fn staging_quota_refusal_restores_a_recycled_predecessor() {
    quota_refusal_keeps_predecessor(true);
}

fn quota_refusal_keeps_predecessor(after_reservation: bool) {
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
    let mut slots = StateSlots::new(Arc::clone(&storage), 1).unwrap();
    let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
    let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
    let output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
    write_constant(host.backend.as_ref(), &input, 2.0);
    write_constant(host.backend.as_ref(), &weights, 1.0);
    let position = LanePosition {
        tokens: 1,
        reserved_tokens: 7,
        ..LanePosition::default()
    };
    slots.begin(0, 0, position).unwrap();
    slots
        .recording(0, 0)
        .unwrap()
        .causal_convolution(0, &input, &weights, &output)
        .unwrap();
    slots.take_batch(0).unwrap().submit_and_finish().unwrap();
    let sequence = slots.sequence_handle(0).unwrap().unwrap();
    let checkpoint = sequence.checkpoint_latest().unwrap();
    refuse_next_recording(&host, &mut slots, position, after_reservation);
    assert_eq!(slots.position(0).unwrap(), checkpoint.position());
    assert_eq!(slots.lanes[0].serial, if after_reservation { 2 } else { 1 });
    sequence.rewind(0).unwrap();
    assert_eq!(
        sequence.position(),
        LanePosition {
            reserved_tokens: 7,
            ..LanePosition::default()
        }
    );
    slots.begin(1, 0, position).unwrap();
    slots
        .recording(1, 0)
        .unwrap()
        .causal_convolution(0, &input, &weights, &output)
        .unwrap();
    slots.take_batch(1).unwrap().submit_and_finish().unwrap();
    assert_eq!(
        host.backend.read(&output).unwrap(),
        2_f32.to_ne_bytes().repeat(7)
    );
}

fn refuse_next_recording(
    host: &Host<forja_cpu::CpuBackend>,
    slots: &mut StateSlots<forja_cpu::CpuBackend>,
    position: LanePosition,
    after_reservation: bool,
) {
    use std::sync::atomic::Ordering;
    let before = host.live_bytes.load(Ordering::Acquire);
    let next = LanePosition {
        tokens: 2,
        ..position
    };
    if after_reservation {
        slots.begin(1, 0, next).unwrap();
    }
    let live = host.live_bytes.load(Ordering::Acquire);
    let budget = super::super::StateCharge::reserve(
        host.backend.as_ref(),
        Arc::clone(&host.live_bytes),
        65536,
        65536 - live,
    )
    .unwrap();
    if after_reservation {
        assert!(matches!(
            super::super::StateCharge::reserve(
                host.backend.as_ref(),
                Arc::clone(&host.live_bytes),
                65536,
                4
            ),
            Err(crate::compute::Error::Quota(_))
        ));
        slots.cancel_unsubmitted_bank(1).unwrap();
    } else {
        assert!(matches!(
            slots.begin(1, 0, next),
            Err(crate::compute::Error::Quota(_))
        ));
    }
    drop(budget);
    assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
}
