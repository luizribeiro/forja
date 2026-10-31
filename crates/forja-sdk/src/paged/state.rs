use super::PagedCacheSet;
use crate::{Error, FloatElement, Result, Tensor};

/// A lane-local capability for one protected serving plane.
pub struct PagedState<'a> {
    pub(super) set: &'a PagedCacheSet,
    pub(super) lane: u32,
    pub(super) plane: u32,
}

#[derive(Clone, Copy)]
enum Update {
    Convolution,
    Recurrent,
    Block,
}

impl PagedState<'_> {
    /// Records raw convolution and carries its private pre-convolution history.
    ///
    /// # Errors
    /// Refuses invalid tensors or a host-denied plane update.
    pub fn causal_convolution<T: FloatElement>(
        &self,
        input: &Tensor<T>,
        weights: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        let mut output = Tensor::<T>::zeros(input.shape())?;
        self.causal_convolution_into(input, weights, &mut output)?;
        Ok(output)
    }

    /// Records one recurrent update using Q/K, values, decay and beta.
    ///
    /// # Errors
    /// Refuses invalid tensors or a host-denied plane update.
    pub fn delta_rule<T: FloatElement>(
        &self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<T>,
        decay: &Tensor<f32>,
        beta: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        let mut output = Tensor::<T>::zeros(v.shape())?;
        self.delta_rule_into(q, k, v, decay, beta, &mut output)?;
        Ok(output)
    }

    /// Records convolution into a caller-owned output view.
    ///
    /// # Errors
    /// Refuses invalid shapes, aliases or a host-denied update.
    pub fn causal_convolution_into<T: FloatElement>(
        &self,
        input: &Tensor<T>,
        weights: &Tensor<T>,
        output: &mut Tensor<T>,
    ) -> Result<()> {
        self.record_into(
            Update::Convolution,
            &[input.handle(), weights.handle()],
            output,
            "",
        )
    }

    /// Records a recurrent update into a caller-owned output view.
    ///
    /// # Errors
    /// Refuses invalid shapes, aliases or a host-denied update.
    pub fn delta_rule_into<T: FloatElement>(
        &self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<T>,
        decay: &Tensor<f32>,
        beta: &Tensor<T>,
        output: &mut Tensor<T>,
    ) -> Result<()> {
        self.record_into(
            Update::Recurrent,
            &[
                q.handle(),
                k.handle(),
                v.handle(),
                decay.handle(),
                beta.handle(),
            ],
            output,
            "",
        )
    }

    /// Records canonical delta-rule blocks with the selected reduction class.
    ///
    /// # Errors
    /// Refuses invalid tensors, picks, quota or a duplicate plane update.
    pub fn delta_rule_block<T: FloatElement>(
        &self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<T>,
        decay: &Tensor<f32>,
        beta: &Tensor<T>,
        pick: Option<&crate::target::metal::Variant>,
    ) -> Result<Tensor<T>> {
        self.record(
            Update::Block,
            &[
                q.handle(),
                k.handle(),
                v.handle(),
                decay.handle(),
                beta.handle(),
            ],
            v.shape(),
            pick.map_or("", crate::target::metal::Variant::name),
        )
    }

    fn record<T: FloatElement>(
        &self,
        operation: Update,
        inputs: &[&crate::sys::Handle],
        shape: &[u32],
        algorithm: &str,
    ) -> Result<Tensor<T>> {
        let output = Tensor::<T>::zeros(shape)?;
        self.record_into(operation, inputs, &output, algorithm)?;
        Ok(output)
    }

    fn record_into<T: FloatElement>(
        &self,
        operation: Update,
        inputs: &[&crate::sys::Handle],
        output: &Tensor<T>,
        algorithm: &str,
    ) -> Result<()> {
        #[cfg(target_family = "wasm")]
        {
            use crate::sys::guest::paged_storage::{self, StateOperation};
            let operation = match operation {
                Update::Convolution => StateOperation::CausalConvolution,
                Update::Recurrent => StateOperation::DeltaRule,
                Update::Block => StateOperation::DeltaRuleBlock,
            };
            crate::graph::record_import(|commands| {
                paged_storage::record_state(
                    commands,
                    &self.set._slot,
                    self.lane,
                    self.plane,
                    operation,
                    inputs,
                    output.handle(),
                    algorithm,
                )
                .map_err(|error| Error::loading(format!("{error:?}")))
            })
        }
        #[cfg(not(target_family = "wasm"))]
        {
            let _ = (
                self.set, self.lane, self.plane, operation, inputs, output, algorithm,
            );
            Err(Error::loading("slotted state requires a component host"))
        }
    }
}
