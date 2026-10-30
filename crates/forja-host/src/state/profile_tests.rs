use super::*;
use compute::HostStatePlane;

#[tokio::test]
async fn metal_state_imports_contribute_scoped_composite_and_stage_costs() {
    let mut host = Host::new(
        forja_metal::MetalBackend::new().unwrap(),
        Limits::new(1_048_576, 8, 1_048_576, 128, 1_048_576),
    );
    let state = host
        .create(compute::Dtype::F32, vec![1, 1, 1])
        .await
        .unwrap()
        .unwrap();
    let mut inputs = Vec::new();
    for shape in [
        vec![2, 1, 1],
        vec![2, 1, 1],
        vec![2, 1, 1],
        vec![2, 1],
        vec![2, 1],
    ] {
        inputs.push(host.alloc(compute::Dtype::F32, &shape).unwrap());
    }
    let output = host.alloc(compute::Dtype::F32, &[2, 1, 1]).unwrap();
    host.begin_profile_step();
    let commands = host.command_list().unwrap();
    compute::HostCommandList::set_profile_scope(
        &mut host,
        commands,
        "mixer".into(),
        "recurrent".into(),
        Some(7),
    )
    .await
    .unwrap()
    .unwrap();
    for profiled in [true, false] {
        host.delta_rule_block(
            Resource::new_borrow(state.rep()),
            inputs
                .iter()
                .map(|input| Resource::new_borrow(input.rep()))
                .collect(),
            Resource::new_borrow(output.rep()),
            "delta-rule-block.wy64-mma-v32".into(),
        )
        .await
        .unwrap()
        .unwrap();
        if profiled {
            let profile = host.finish_profile_step(Duration::from_secs(1)).unwrap();
            assert_eq!(profile.submissions.len(), 1);
            let dispatch = &profile.submissions[0].per_dispatch[0];
            assert_eq!(dispatch.scope.as_ref().unwrap().layer(), Some(7));
            assert_eq!(dispatch.scope.as_ref().unwrap().layer_type(), "recurrent");
            assert_eq!(dispatch.internal.len(), 4);
            let total = dispatch
                .internal
                .iter()
                .try_fold(forja_core::KernelCost::default(), |sum, stage| {
                    sum.checked_add(stage.cost.unwrap())
                })
                .unwrap();
            assert_eq!(total, dispatch.cost);
            assert_eq!(total.state_read_bytes, 4);
            assert!(total.scratch_read_bytes > 0);
        } else {
            assert!(host.state_cell(&state).unwrap().take_profile().is_none());
        }
    }
}
