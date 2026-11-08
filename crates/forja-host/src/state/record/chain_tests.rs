use super::{test_support::Fixture, *};
use crate::StateBatch;

fn chained_updates<B: Backend + Send + Sync + 'static>(backend: B) {
    let f = Fixture::new(backend);
    let mut chain_lane = f.reserve();
    let mut plain_lane = f.reserve();
    let mut chain = f.commands(&mut chain_lane, 1, 2);
    f.update(&mut chain, 0);
    assert!(!chain.updates_complete());
    f.update(&mut chain, 1);
    assert!(chain.updates_complete());
    for (index, dispatch) in chain.commands.dispatches().iter().enumerate() {
        let Op::State(operation) = dispatch.op() else {
            unreachable!()
        };
        assert_eq!(operation.access(), chain.access.step(index / 2).unwrap());
        if index >= 2 {
            let suffix = operation.input_suffix(dispatch.inputs()).unwrap();
            assert_eq!(
                suffix[0],
                chain.commands.dispatches()[index - 2].outputs()[1]
            );
        }
    }
    let versions = std::iter::once(chain.access.source())
        .chain(chain.access.destinations())
        .collect::<Vec<_>>();
    f.execute(chain);
    let expected = f.read_outputs();
    for (row, expected_row) in expected.iter().enumerate() {
        let mut plain = f.commands(&mut plain_lane, u64::try_from(row + 1).unwrap(), 1);
        f.update(&mut plain, row);
        let records = plain.access.destination();
        f.execute(plain);
        for (plane, expected_output) in expected_row.iter().enumerate() {
            assert_eq!(
                f.host.backend.read(&f.outputs[row][plane]).unwrap(),
                *expected_output
            );
            assert_eq!(
                f.host
                    .backend
                    .read(&f.storage.version_view(plane, records).unwrap())
                    .unwrap(),
                f.host
                    .backend
                    .read(&f.storage.version_view(plane, versions[row + 1]).unwrap())
                    .unwrap(),
            );
        }
    }
}

#[test]
fn chained_state_matches_separate_updates() {
    chained_updates(forja_cpu::CpuBackend::new());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_chained_state_matches_separate_updates() {
    chained_updates(forja_metal::MetalBackend::new().unwrap());
}

#[test]
fn refused_updates_preserve_the_next_link() {
    let f = Fixture::new(forja_cpu::CpuBackend::new());
    let mut lane = f.reserve();
    let mut chain = f.commands(&mut lane, 1, 2);
    let prepare = |commands: &StateCommands<_>, access| {
        commands.prepare_update(
            0,
            StateOp::CausalDepthwiseConv(access),
            &[&f.input, &f.weights],
            &f.outputs[0][0],
            None,
        )
    };
    assert!(prepare(&chain, chain.access.step(1).unwrap()).is_err());
    assert!(
        chain
            .causal_convolution(0, &f.input, &f.weights, &f.input)
            .is_err()
    );
    assert!(
        chain
            .causal_convolution(2, &f.input, &f.weights, &f.outputs[0][0])
            .is_err()
    );
    let first = prepare(&chain, chain.access.step(0).unwrap()).unwrap();
    let limit = chain.limits.work_per_dispatch;
    chain.limits.work_per_dispatch = 0;
    assert!(chain.record_update(0, first.clone()).is_err());
    assert_eq!(chain.updated, [0, 0]);
    chain.limits.work_per_dispatch = limit;
    assert!(chain.record_update(usize::MAX, first.clone()).is_err());
    chain.record_update(0, first.clone()).unwrap();
    assert!(chain.record_update(0, first).is_err());
    let second = prepare(&chain, chain.access.step(1).unwrap()).unwrap();
    chain.record_update(0, second).unwrap();
    assert!(
        chain
            .causal_convolution(0, &f.input, &f.weights, &f.outputs[0][0])
            .is_err()
    );
    assert_eq!(chain.updated, [2, 0]);
    assert!(!chain.updates_complete());
}

#[test]
fn submissions_require_exactly_the_chain_length_on_every_plane() {
    let f = Fixture::new(forja_cpu::CpuBackend::new());
    let mut lane = f.reserve();
    let chain = f.commands(&mut lane, 1, 2);
    let sequence = f.storage.reserve().unwrap();
    let mut serial = 0;
    for batch in [false, true] {
        for counts in [[0, 0], [1, 1], [2, 1], [3, 2]] {
            serial += 1;
            let mut recording = f
                .storage
                .record(&sequence, sequence.begin(serial).unwrap())
                .unwrap();
            recording.access = chain.access.clone();
            recording.updated = counts.to_vec();
            if batch {
                assert!(StateBatch::new(vec![recording]).is_err());
            } else {
                assert!(recording.submit().is_err());
            }
        }
    }
}

fn corrupted_link<B: Backend + 'static>(backend: impl Fn() -> B) {
    for table in [false, true] {
        let f = Fixture::new(backend());
        let mut lane = f.reserve();
        let mut chain = f.commands(&mut lane, 1, 2);
        f.update(&mut chain, 0);
        f.update(&mut chain, 1);
        if table {
            let first = f.host.backend.read(&chain.table_view(0).unwrap()).unwrap();
            f.host
                .backend
                .write(&chain.table_view(1).unwrap(), &first)
                .unwrap();
        } else {
            let mut generations = f.host.backend.read(&chain.metadata[1]).unwrap();
            let offset = usize::try_from(chain.access.destination()).unwrap() * 4;
            generations[offset..offset + 4].fill(0);
            f.host
                .backend
                .write(&chain.metadata[1], &generations)
                .unwrap();
        }
        assert!(f.execute_result(chain).is_err());
    }
}

#[test]
fn chained_updates_refuse_corrupt_later_authentication() {
    corrupted_link(forja_cpu::CpuBackend::new);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_chained_updates_refuse_corrupt_later_authentication() {
    corrupted_link(|| forja_metal::MetalBackend::new().unwrap());
}

#[test]
fn missing_authentication_metadata_is_refused() {
    let f = Fixture::new(forja_cpu::CpuBackend::new());
    let mut lane = f.reserve();
    let mut chain = f.commands(&mut lane, 1, 2);
    Arc::get_mut(&mut chain.metadata).unwrap().tensors.clear();
    assert!(chain.table_view(0).is_err());
    assert!(
        chain
            .causal_convolution(0, &f.input, &f.weights, &f.outputs[0][0])
            .is_err()
    );
    assert_eq!(chain.updated, [0, 0]);
}
