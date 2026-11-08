use super::*;
use crate::{Host, Limits, StateAction};
use forja_core::{DType, StateLayout, StatePlane};

fn chain_bindings<B: Backend + Send + Sync + 'static>(backend: B) {
    let host = Host::new(backend, Limits::new(1_048_576, 8, 4096, 128, 4096));
    for rows in [2, 3, 7] {
        let storage = host
            .create_state_storage_with_spares(
                StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap(),
                6,
                rows - 2,
            )
            .unwrap();
        let mut slots = StateSlots::new(Arc::clone(&storage), 6).unwrap();
        assert!(slots.trace_chain(0).is_err());
        assert!(slots.inactive_chain(rows + 1).is_err());
        let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
        let output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        host.backend
            .write(&input, &2_f32.to_le_bytes().repeat(7))
            .unwrap();
        host.backend
            .write(&weights, &1_f32.to_le_bytes().repeat(28))
            .unwrap();
        let mut trace = slots.trace_chain(rows).unwrap();
        for _ in 0..rows {
            trace
                .lane(5)
                .unwrap()
                .causal_convolution(0, &input, &weights, &output)
                .unwrap();
        }
        let capture = trace.lane(5).unwrap().capture().unwrap();
        let template = std::mem::take(&mut trace.lane(5).unwrap().commands);
        assert_eq!(slots.commands(0, 0, false).unwrap().access.chain_len(), 1);
        slots.admit(5).unwrap();
        let sequence = slots.sequence_handle(5).unwrap().unwrap();
        assert!(sequence.try_reserve_spares().unwrap());
        let position = LanePosition {
            tokens: u64::from(rows),
            reserved_tokens: 33,
            ..LanePosition::default()
        };
        slots
            .begin_action(0, 5, position, Some(StateAction::Verify))
            .unwrap();
        assert!(slots.begin(0, 1, position).is_err());
        let padding = slots.commands(0, 0, false).unwrap();
        assert_eq!(padding.access.chain_len(), usize::try_from(rows).unwrap());
        for row in 0..usize::try_from(rows).unwrap() {
            assert!(!padding.access.step(row).unwrap().is_active());
        }
        assert!(
            host.backend
                .read(&padding.metadata[0])
                .unwrap()
                .iter()
                .all(|byte| *byte == u8::MAX)
        );
        let recording = slots.recording(0, 5).unwrap();
        let transaction = recording.transaction.clone();
        let bindings = capture.bind(recording).unwrap();
        recording.commands = bindings.apply(template).unwrap();
        slots.take_batch(0).unwrap().submit_and_finish().unwrap();
        assert!(slots.release(5).is_err());
        assert_eq!(sequence.position().tokens, 0);
        sequence
            .publish_accepted(&transaction, rows.checked_sub(1).unwrap())
            .unwrap();
        assert_eq!(sequence.position().tokens, u64::from(rows));
        assert_eq!(
            host.backend.read(&output).unwrap(),
            (f32::from(u16::try_from(rows.min(4)).unwrap()) * 2.)
                .to_le_bytes()
                .repeat(7)
        );
        slots
            .begin(
                0,
                5,
                LanePosition {
                    tokens: u64::from(rows) + 1,
                    ..position
                },
            )
            .unwrap();
        assert_eq!(slots.commands(0, 0, false).unwrap().access.chain_len(), 1);
        slots.cancel_unsubmitted_bank(0).unwrap();
        slots.release(5).unwrap();
    }
}

#[test]
fn verify_slot_trace_and_padding_follow_the_active_chain() {
    chain_bindings(forja_cpu::CpuBackend::new());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_verify_slot_trace_and_padding_follow_the_active_chain() {
    chain_bindings(forja_metal::MetalBackend::new().unwrap());
}
