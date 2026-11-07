use std::{
    sync::{
        Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};

use block2::RcBlock;
use forja_core::BackendError;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineReflection, MTLComputePipelineState, MTLDevice,
    MTLFunction, MTLFunctionConstantValues, MTLLibrary, MTLPipelineOption,
};

enum Artifact {
    Library(Retained<ProtocolObject<dyn MTLLibrary>>),
    Function(Retained<ProtocolObject<dyn MTLFunction>>),
    Pipeline(Retained<ProtocolObject<dyn MTLComputePipelineState>>),
}

// SAFETY: Compiled Metal libraries, functions and pipeline states are immutable and may be
// shared across host threads. Callback ownership transfers exactly one retained reference.
unsafe impl Send for Artifact {}

pub(super) struct Compiler {
    gate: Mutex<()>,
    remaining: Mutex<Duration>,
    disabled: AtomicBool,
}

impl Compiler {
    pub(super) fn new(budget: Duration) -> Result<Self, BackendError> {
        if budget.is_zero() || Instant::now().checked_add(budget).is_none() {
            return Err(BackendError::InvalidInput);
        }
        Ok(Self {
            gate: Mutex::new(()),
            remaining: Mutex::new(budget),
            disabled: AtomicBool::new(false),
        })
    }

    pub(super) fn ensure_enabled(&self) -> Result<(), BackendError> {
        if self.disabled.load(Ordering::Acquire) {
            Err(BackendError::Timeout)
        } else {
            Ok(())
        }
    }

    #[cfg(test)]
    pub(super) fn expire_for_test(&self) {
        self.disabled.store(true, Ordering::Release);
    }

    fn wait<T: Send>(
        &self,
        start: impl FnOnce(mpsc::SyncSender<Result<T, BackendError>>),
    ) -> Result<T, BackendError> {
        let _gate = self
            .gate
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        self.ensure_enabled()?;
        let allowance = *self
            .remaining
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)?;
        if allowance.is_zero() {
            self.disabled.store(true, Ordering::Release);
            return Err(BackendError::Timeout);
        }
        let started = Instant::now();
        let (sender, receiver) = mpsc::sync_channel(1);
        start(sender);
        let result = receiver.recv_timeout(allowance.saturating_sub(started.elapsed()));
        let left = allowance.saturating_sub(started.elapsed());
        *self
            .remaining
            .lock()
            .map_err(|_| BackendError::ExecutionFailed)? = left;
        if left.is_zero() || matches!(result, Err(mpsc::RecvTimeoutError::Timeout)) {
            self.disabled.store(true, Ordering::Release);
            return Err(BackendError::Timeout);
        }
        result.map_err(|_| BackendError::ExecutionFailed)?
    }

    pub(super) fn library(
        &self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        source: &Retained<NSString>,
        options: &Retained<MTLCompileOptions>,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, BackendError> {
        let result = self.wait(|sender| {
            let inputs = (device.clone(), source.clone(), options.clone());
            let callback = RcBlock::new(
                move |value: *mut ProtocolObject<dyn MTLLibrary>, _: *mut NSError| {
                    let _inputs = &inputs;
                    // SAFETY: Metal lends a live result until the callback returns; retaining transfers
                    // its lifetime to the waiting thread. A null result denotes compilation failure.
                    let value = unsafe { Retained::retain(value) }
                        .map(Artifact::Library)
                        .ok_or(BackendError::ExecutionFailed);
                    let _ = sender.try_send(value);
                },
            );
            // SAFETY: The heap block is live for this call and Metal copies it for asynchronous use.
            unsafe {
                device.newLibraryWithSource_options_completionHandler(
                    source,
                    Some(options),
                    RcBlock::as_ptr(&callback),
                );
            }
        })?;
        match result {
            Artifact::Library(value) => Ok(value),
            _ => Err(BackendError::ExecutionFailed),
        }
    }

    pub(super) fn function(
        &self,
        library: &Retained<ProtocolObject<dyn MTLLibrary>>,
        name: &Retained<NSString>,
        values: &Retained<MTLFunctionConstantValues>,
    ) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, BackendError> {
        let result = self.wait(|sender| {
            let inputs = (library.clone(), name.clone(), values.clone());
            let callback = RcBlock::new(
                move |value: *mut ProtocolObject<dyn MTLFunction>, _: *mut NSError| {
                    let _inputs = &inputs;
                    // SAFETY: Metal lends a live immutable result, or null on failure.
                    let value = unsafe { Retained::retain(value) }
                        .map(Artifact::Function)
                        .ok_or(BackendError::ExecutionFailed);
                    let _ = sender.try_send(value);
                },
            );
            library.newFunctionWithName_constantValues_completionHandler(name, values, &callback);
        })?;
        match result {
            Artifact::Function(value) => Ok(value),
            _ => Err(BackendError::ExecutionFailed),
        }
    }

    pub(super) fn pipeline(
        &self,
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
        function: &Retained<ProtocolObject<dyn MTLFunction>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, BackendError> {
        let result = self.wait(|sender| {
            let inputs = (device.clone(), function.clone());
            let callback = RcBlock::new(
                move |value: *mut ProtocolObject<dyn MTLComputePipelineState>,
                      _: *mut MTLComputePipelineReflection,
                      _: *mut NSError| {
                    let _inputs = &inputs;
                    // SAFETY: Metal lends a live immutable result, or null on failure.
                    let value = unsafe { Retained::retain(value) }
                        .map(Artifact::Pipeline)
                        .ok_or(BackendError::ExecutionFailed);
                    let _ = sender.try_send(value);
                },
            );
            // SAFETY: Metal copies the live heap block; reflection is not requested or accessed.
            unsafe {
                device.newComputePipelineStateWithFunction_options_completionHandler(
                    function,
                    MTLPipelineOption::None,
                    RcBlock::as_ptr(&callback),
                );
            }
        })?;
        match result {
            Artifact::Pipeline(value) => Ok(value),
            _ => Err(BackendError::ExecutionFailed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timeout_disables_further_compiler_work() {
        let compiler = Compiler::new(Duration::from_millis(10)).unwrap();
        let mut retained_sender = None;
        assert_eq!(
            compiler.wait::<u8>(|sender| retained_sender = Some(sender)),
            Err(BackendError::Timeout)
        );
        assert_eq!(
            compiler.wait::<u8>(|_| panic!("disabled compiler started another job")),
            Err(BackendError::Timeout)
        );
        drop(retained_sender);
    }

    #[test]
    fn cumulative_allowance_covers_success_and_failure() {
        let compiler = Compiler::new(Duration::from_secs(60)).unwrap();
        assert_eq!(
            compiler.wait(|sender| {
                sender.try_send(Ok(7_u8)).unwrap();
            }),
            Ok(7)
        );
        assert_eq!(
            compiler.wait::<u8>(|sender| {
                sender.try_send(Err(BackendError::ExecutionFailed)).unwrap();
            }),
            Err(BackendError::ExecutionFailed)
        );
        *compiler.remaining.lock().unwrap() = Duration::ZERO;
        assert_eq!(
            compiler.wait::<u8>(|_| panic!("exhausted compiler started another job")),
            Err(BackendError::Timeout)
        );
        assert!(Compiler::new(Duration::ZERO).is_err());
        assert!(Compiler::new(Duration::MAX).is_err());
    }
}
