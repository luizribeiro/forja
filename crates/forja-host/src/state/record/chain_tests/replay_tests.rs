use super::*;

fn rebind_chain<B: Backend + Send + Sync + 'static>(backend: B) {
    let f = Fixture::new(backend);
    let mut original_lane = f.reserve();
    let mut chain = f.commands(&mut original_lane, 1, 2);
    f.update(&mut chain, 0);
    f.update(&mut chain, 1);
    let capture = chain.capture().unwrap();
    capture.bind(&mut chain).unwrap();
    assert_eq!(chain.updated, [2, 2]);
    let template = chain.commands.clone();
    let source = chain.access.source();
    f.execute(chain);
    let expected = f.read_outputs();

    let mut other_lane = f.reserve();
    let mut shorter = f.commands(&mut other_lane, 1, 1);
    assert!(capture.bind(&mut shorter).is_err());
    assert_eq!(shorter.updated, [0, 0]);
    drop(shorter);
    drop(other_lane);
    let mut rebound_lane = f.reserve();
    let mut rebound = f.commands(&mut rebound_lane, 1, 2);
    assert_ne!(rebound.access.source(), source);
    let bindings = capture.bind(&mut rebound).unwrap();
    assert_eq!(rebound.updated, [2, 2]);
    assert!(capture.bind(&mut rebound).is_err());
    rebound.commands = bindings.apply(template).unwrap();
    for (index, dispatch) in rebound.commands.dispatches().iter().enumerate() {
        let Op::State(operation) = dispatch.op() else {
            unreachable!()
        };
        assert_eq!(operation.access(), rebound.access.step(index / 2).unwrap());
    }
    f.execute(rebound);
    assert_eq!(f.read_outputs(), expected);
}

#[test]
fn captured_chains_rebind_every_link_without_double_counting() {
    rebind_chain(forja_cpu::CpuBackend::new());
}

#[cfg(target_os = "macos")]
#[test]
fn metal_captured_chains_rebind_every_link_without_double_counting() {
    rebind_chain(forja_metal::MetalBackend::new().unwrap());
}
