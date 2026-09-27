use std::cell::RefCell;

use crate::{Result, sys};

thread_local! {
    static CURRENT: RefCell<Option<sys::Commands>> = const { RefCell::new(None) };
}

pub(crate) fn record(
    operation: sys::Op,
    inputs: &[&sys::Handle],
    output: &sys::Handle,
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            *current = Some(sys::command_list()?);
        }
        let commands = current
            .as_mut()
            .ok_or_else(|| crate::Error::new("current graph was not initialized"))?;
        sys::dispatch(commands, operation, inputs, output)
    })
}

pub(crate) fn record_program(
    program: sys::Program,
    inputs: &[&sys::Handle],
    outputs: &[&sys::Handle],
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            *current = Some(sys::command_list()?);
        }
        let commands = current
            .as_mut()
            .ok_or_else(|| crate::Error::new("current graph was not initialized"))?;
        sys::dispatch_program(commands, program, inputs, outputs)
    })
}

/// Submits all operations recorded by the current thread.
///
/// # Errors
///
/// Returns a host validation or execution error.
pub fn eval() -> Result<()> {
    CURRENT.with(|current| match current.borrow_mut().take() {
        Some(commands) => sys::submit(commands),
        None => Ok(()),
    })
}
