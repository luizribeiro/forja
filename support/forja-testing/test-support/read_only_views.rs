//! Shared checks for lending reduced-access tensor capabilities.
#![allow(clippy::unwrap_used)]
use forja_core::{Backend, BackendError, CommandList, DType, Op, OpError, Submission, ViewOp};

/// Checks that views preserve reduced rights while the owner remains writable.
pub fn check(backend: &impl Backend) {
    let owner = backend.alloc(DType::F32, &[7, 33]).unwrap();
    let bytes = 1.0_f32.to_ne_bytes().repeat(7 * 33);
    backend.write(&owner, &bytes).unwrap();
    let borrowed = owner.read_only();
    let view = backend
        .view(&borrowed, ViewOp::Reshape(vec![33, 7]))
        .unwrap();
    assert!(!view.is_writable());
    assert_eq!(backend.read(&view).unwrap(), bytes);
    for tensor in [&borrowed, &view] {
        assert_eq!(
            backend.write(tensor, &bytes),
            Err(BackendError::InvalidInput)
        );
        assert_eq!(
            CommandList::new().dispatch(Op::Copy, &[&owner], tensor),
            Err(OpError::ReadOnlyOutput)
        );
    }
    let output = backend.alloc(DType::F32, &[33, 7]).unwrap();
    let mut commands = CommandList::new();
    commands.dispatch(Op::Copy, &[&view], &output).unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    assert_eq!(backend.read(&output).unwrap(), bytes);
    let replacement = 2.0_f32.to_ne_bytes().repeat(7 * 33);
    backend.write(&owner, &replacement).unwrap();
    assert_eq!(backend.read(&borrowed).unwrap(), replacement);
    backend.release(&output).unwrap();
    backend.release(&owner).unwrap();
}
