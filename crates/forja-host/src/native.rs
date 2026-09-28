use std::{sync::Arc, time::Duration};

use forja_core::{
    Backend, BackendError, CommandList, DType, Op, Submission, Tensor, ViewOp,
    program::{KernelSignature, PreparedProgram, ValidatedProgram, prepare_program},
};

/// An in-process host used by trusted native development and tests.
#[derive(Debug)]
pub struct NativeHost<B: Backend> {
    backend: Arc<B>,
}

impl<B: Backend> NativeHost<B> {
    /// Wraps a backend for direct in-process use.
    #[must_use]
    pub fn new(backend: B) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    /// Allocates a contiguous tensor initialized to zero.
    ///
    /// # Errors
    ///
    /// Returns the backend allocation failure.
    pub fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<NativeTensor<B>, BackendError> {
        let tensor = self.backend.alloc(dtype, shape)?;
        Ok(NativeTensor::new(Arc::clone(&self.backend), tensor))
    }

    /// Creates a validated metadata-only view.
    ///
    /// # Errors
    ///
    /// Returns an error when the view is invalid or belongs to another host.
    pub fn view(
        &self,
        tensor: &NativeTensor<B>,
        operation: ViewOp,
    ) -> Result<NativeTensor<B>, BackendError> {
        if !Arc::ptr_eq(&self.backend, &tensor.allocation.backend) {
            return Err(BackendError::InvalidInput);
        }
        let view = self.backend.view(&tensor.tensor, operation)?;
        Ok(NativeTensor {
            tensor: view,
            allocation: Arc::clone(&tensor.allocation),
        })
    }

    /// Writes contiguous logical tensor bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for foreign tensors or invalid writes.
    pub fn write(&self, tensor: &NativeTensor<B>, bytes: &[u8]) -> Result<(), BackendError> {
        self.validate(tensor)?;
        self.backend.write(&tensor.tensor, bytes)
    }

    /// Gathers a tensor view into contiguous logical bytes.
    ///
    /// # Errors
    ///
    /// Returns an error for a foreign tensor or backend read failure.
    pub fn read(&self, tensor: &NativeTensor<B>) -> Result<Vec<u8>, BackendError> {
        self.validate(tensor)?;
        self.backend.read(&tensor.tensor)
    }

    /// Creates an empty command list for this host.
    #[must_use]
    pub fn command_list(&self) -> NativeCommandList<B> {
        NativeCommandList {
            backend: Arc::clone(&self.backend),
            commands: CommandList::new(),
            retained: Vec::new(),
        }
    }

    /// Validates and prepares a scalar program for repeated dispatch.
    ///
    /// # Errors
    ///
    /// Returns an error when validation or backend preparation fails.
    pub fn prepare_program(
        &self,
        program: ValidatedProgram,
        signature: KernelSignature,
    ) -> Result<NativeKernel<B>, BackendError> {
        let program = prepare_program(self.backend.as_ref(), program, signature)
            .map_err(|_| BackendError::InvalidInput)?;
        Ok(NativeKernel {
            backend: Arc::clone(&self.backend),
            program,
        })
    }

    fn validate(&self, tensor: &NativeTensor<B>) -> Result<(), BackendError> {
        Arc::ptr_eq(&self.backend, &tensor.allocation.backend)
            .then_some(())
            .ok_or(BackendError::InvalidInput)
    }
}

/// A prepared scalar program owned by one native host.
#[derive(Debug)]
pub struct NativeKernel<B: Backend> {
    backend: Arc<B>,
    program: Arc<PreparedProgram>,
}

/// A refcounted native tensor handle.
#[derive(Debug)]
pub struct NativeTensor<B: Backend> {
    tensor: Tensor,
    allocation: Arc<NativeAllocation<B>>,
}

impl<B: Backend> NativeTensor<B> {
    fn new(backend: Arc<B>, tensor: Tensor) -> Self {
        Self {
            allocation: Arc::new(NativeAllocation {
                backend,
                owner: tensor.clone(),
            }),
            tensor,
        }
    }
}

impl<B: Backend> Clone for NativeTensor<B> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            allocation: Arc::clone(&self.allocation),
        }
    }
}

#[derive(Debug)]
struct NativeAllocation<B: Backend> {
    backend: Arc<B>,
    owner: Tensor,
}

impl<B: Backend> Drop for NativeAllocation<B> {
    fn drop(&mut self) {
        let _ = self.backend.release(&self.owner);
    }
}

/// A native command list retaining every referenced tensor allocation.
#[derive(Debug)]
pub struct NativeCommandList<B: Backend> {
    backend: Arc<B>,
    commands: CommandList,
    retained: Vec<NativeTensor<B>>,
}

impl<B: Backend> NativeCommandList<B> {
    /// Validates and records one operation.
    ///
    /// # Errors
    ///
    /// Returns an error for foreign tensors or an invalid operation signature.
    pub fn dispatch(
        &mut self,
        operation: Op,
        inputs: &[&NativeTensor<B>],
        output: &NativeTensor<B>,
    ) -> Result<(), BackendError> {
        if inputs
            .iter()
            .chain(std::iter::once(&output))
            .any(|tensor| !Arc::ptr_eq(&self.backend, &tensor.allocation.backend))
        {
            return Err(BackendError::InvalidInput);
        }
        self.commands
            .dispatch(
                operation,
                &inputs
                    .iter()
                    .map(|tensor| &tensor.tensor)
                    .collect::<Vec<_>>(),
                &output.tensor,
            )
            .map_err(|_| BackendError::InvalidInput)?;
        self.retained
            .extend(inputs.iter().map(|tensor| (*tensor).clone()));
        self.retained.push(output.clone());
        Ok(())
    }

    /// Validates and records one prepared scalar program.
    ///
    /// # Errors
    ///
    /// Returns an error for foreign resources or an invalid program binding.
    pub fn dispatch_kernel(
        &mut self,
        kernel: &NativeKernel<B>,
        inputs: &[&NativeTensor<B>],
        outputs: &[&NativeTensor<B>],
    ) -> Result<(), BackendError> {
        if !Arc::ptr_eq(&self.backend, &kernel.backend)
            || inputs
                .iter()
                .chain(outputs)
                .any(|tensor| !Arc::ptr_eq(&self.backend, &tensor.allocation.backend))
        {
            return Err(BackendError::InvalidInput);
        }
        self.commands
            .dispatch_kernel(
                &kernel.program,
                &inputs
                    .iter()
                    .map(|tensor| &tensor.tensor)
                    .collect::<Vec<_>>(),
                &outputs
                    .iter()
                    .map(|tensor| &tensor.tensor)
                    .collect::<Vec<_>>(),
            )
            .map_err(|_| BackendError::InvalidInput)?;
        self.retained
            .extend(inputs.iter().map(|tensor| (*tensor).clone()));
        self.retained
            .extend(outputs.iter().map(|tensor| (*tensor).clone()));
        Ok(())
    }

    /// Submits the recorded work and waits for completion.
    ///
    /// # Errors
    ///
    /// Returns a backend submission or execution failure.
    pub fn submit(self) -> Result<Option<Duration>, BackendError> {
        let submission = self.backend.submit(self.commands)?;
        submission.wait()?;
        Ok(submission.gpu_time())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::Slice;

    #[test]
    fn views_share_an_allocation_until_the_last_handle_drops() {
        let host = NativeHost::new(forja_cpu::CpuBackend::new());
        let tensor = host.alloc(DType::U32, &[2, 3]).unwrap();
        host.write(
            &tensor,
            &(0_u32..6).flat_map(u32::to_le_bytes).collect::<Vec<_>>(),
        )
        .unwrap();
        let view = host
            .view(
                &tensor,
                ViewOp::Slice(vec![
                    Slice::new(0, 2, 1).unwrap(),
                    Slice::new(0, 2, 2).unwrap(),
                ]),
            )
            .unwrap();
        drop(tensor);

        let bytes = host.read(&view).unwrap();
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        assert_eq!(values, [0, 2, 3, 5]);
    }
}
