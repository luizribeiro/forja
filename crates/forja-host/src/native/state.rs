use super::{NativeHost, NativeTensor};
use crate::{Limits, StateCell, StateStorage, compute, guest_error, invalid_handle};
use forja_core::{Backend, DType, StateLayout, StatePlane, Tensor};
use std::sync::Arc;

const NATIVE_STATE_HANDLE_LIMIT: usize = 10_000;

fn state_limits() -> Limits {
    Limits::new(
        u64::MAX,
        forja_core::MAX_RANK,
        u64::MAX,
        NATIVE_STATE_HANDLE_LIMIT,
        u64::MAX,
    )
}

/// Opaque native development counterpart of a guest state capability.
pub struct NativeState<B: Backend> {
    backend: Arc<B>,
    cell: StateCell<B>,
}

impl<B: Backend> NativeHost<B> {
    /// Allocates one private, zeroed state plane.
    ///
    /// # Errors
    /// Returns geometry, quota or backend allocation failures.
    pub fn state(&self, dtype: DType, shape: &[u32]) -> Result<NativeState<B>, compute::Error> {
        let plane = StatePlane::new(dtype, shape.to_vec()).map_err(guest_error)?;
        let layout = StateLayout::new(vec![plane]).map_err(guest_error)?;
        let storage = StateStorage::create(
            Arc::clone(&self.backend),
            state_limits(),
            Arc::clone(&self.state_bytes),
            layout,
            1,
        )?;
        Ok(NativeState {
            backend: Arc::clone(&self.backend),
            cell: StateCell::new(storage)?,
        })
    }
}

impl<B: Backend> NativeState<B> {
    fn tensor<'a>(&self, tensor: &'a NativeTensor<B>) -> Result<&'a Tensor, compute::Error> {
        if !Arc::ptr_eq(&self.backend, &tensor.allocation.backend) {
            return Err(invalid_handle("state input belongs to another host"));
        }
        Ok(&tensor.tensor)
    }

    /// Updates recurrent state, publishing only after successful completion.
    ///
    /// # Errors
    /// Refuses foreign tensors, invalid shapes, aliasing or failed execution.
    pub fn delta_rule(
        &mut self,
        inputs: [&NativeTensor<B>; 5],
        output: &NativeTensor<B>,
    ) -> Result<(), compute::Error> {
        let inputs = inputs
            .map(|input| self.tensor(input))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let output = self.tensor(output)?;
        self.cell
            .delta_rule(
                [inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]],
                output,
                state_limits(),
            )
            .map(|_| ())
    }

    /// Advances private convolution history after successful completion.
    ///
    /// # Errors
    /// Refuses foreign tensors, invalid shapes, aliasing or failed execution.
    pub fn causal_convolution(
        &mut self,
        input: &NativeTensor<B>,
        weights: &NativeTensor<B>,
        output: &NativeTensor<B>,
    ) -> Result<(), compute::Error> {
        self.cell
            .causal_convolution(
                self.tensor(input)?,
                self.tensor(weights)?,
                self.tensor(output)?,
                state_limits(),
            )
            .map(|_| ())
    }
}

impl<B: Backend + 'static> NativeState<B> {
    /// Applies canonical block transforms with an explicit Metal catalog identity.
    ///
    /// # Errors
    /// Refuses foreign tensors, invalid geometry, picks, quotas or failed execution.
    pub fn delta_rule_block(
        &mut self,
        inputs: [&NativeTensor<B>; 5],
        output: &NativeTensor<B>,
        variant: &str,
    ) -> Result<(), compute::Error> {
        let inputs = inputs
            .map(|input| self.tensor(input))
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let output = self.tensor(output)?;
        self.cell
            .delta_rule_block(
                [inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]],
                output,
                state_limits(),
                Some(variant),
            )
            .map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn private_history_persists_and_refuses_foreign_tensors() {
        let host = NativeHost::new(forja_cpu::CpuBackend::new());
        let mut state = host.state(DType::F32, &[1, 1]).unwrap();
        let x = host.alloc(DType::F32, &[1, 1]).unwrap();
        let w = host.alloc(DType::F32, &[2, 1]).unwrap();
        let y = host.alloc(DType::F32, &[1, 1]).unwrap();
        host.write(&x, &2_f32.to_le_bytes()).unwrap();
        host.write(&w, &[1_f32.to_le_bytes(), 1_f32.to_le_bytes()].concat())
            .unwrap();
        state.causal_convolution(&x, &w, &y).unwrap();
        assert_eq!(host.read(&y).unwrap(), 2_f32.to_le_bytes());
        state.causal_convolution(&x, &w, &y).unwrap();
        assert_eq!(host.read(&y).unwrap(), 4_f32.to_le_bytes());
        let bad = host.alloc(DType::F32, &[7]).unwrap();
        assert!(state.causal_convolution(&x, &w, &bad).is_err());
        state.causal_convolution(&x, &w, &y).unwrap();
        assert_eq!(host.read(&y).unwrap(), 4_f32.to_le_bytes());
        let other = NativeHost::new(forja_cpu::CpuBackend::new());
        let foreign = other.alloc(DType::F32, &[1, 1]).unwrap();
        assert!(state.causal_convolution(&foreign, &w, &y).is_err());
        assert!(host.state(DType::F32, &[0]).is_err());
    }
}
