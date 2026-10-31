use crate::{DType, KernelCost, OpError, StateAccess, Tensor};

// Activity, two expected identities, disjointness, two bounds and two generations.
const METADATA_INPUTS: usize = 3;
const AUTHENTICATION_SCALAR_OPS: u64 = 1 + 2 + 1 + 2 + 2;

/// Generic operations over an authenticated out-of-place state version.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StateOp {
    /// One gated delta-rule update; inputs are Q, K, V, decay, beta, state and metadata.
    DeltaRule(StateAccess),
    /// Fixed-size WY blocks; inputs use log decay, with private float32 scratch as output three.
    /// Scratch contents after execution are unspecified.
    DeltaRuleBlock(StateAccess),
    /// Causal depthwise convolution; inputs are tokens, tap weights, history and metadata.
    CausalDepthwiseConv(StateAccess),
}

impl StateOp {
    /// Authorized source and destination versions.
    #[must_use]
    pub const fn access(self) -> StateAccess {
        match self {
            Self::DeltaRule(access)
            | Self::DeltaRuleBlock(access)
            | Self::CausalDepthwiseConv(access) => access,
        }
    }

    /// Output index of the private workspace produced by a block update.
    pub const BLOCK_WORKSPACE_OUTPUT: usize = 2;

    /// Appends the source state and authentication metadata in their shared operand order.
    ///
    /// # Errors
    /// Refuses an operation's incorrect data-operand count.
    pub fn bind_inputs<'a>(
        self,
        data: &[&'a Tensor],
        source: &'a Tensor,
        metadata: &'a [Tensor],
    ) -> Result<Vec<&'a Tensor>, OpError> {
        if metadata.len() != METADATA_INPUTS {
            return Err(OpError::Arity {
                expected: METADATA_INPUTS,
                actual: metadata.len(),
            });
        }
        let mut inputs = data.to_vec();
        inputs.push(source);
        inputs.extend(metadata.iter());
        self.input_suffix(&inputs)?;
        Ok(inputs)
    }

    /// Returns source state, authentication table, generations and activity, in that order.
    ///
    /// # Errors
    /// Refuses mismatched operand layouts before locating authentication identities.
    pub fn input_suffix<T>(self, inputs: &[T]) -> Result<&[T; 4], OpError> {
        let expected = match self {
            Self::DeltaRule(_) | Self::DeltaRuleBlock(_) => 9,
            Self::CausalDepthwiseConv(_) => 6,
        };
        if inputs.len() != expected {
            return Err(OpError::Arity {
                expected,
                actual: inputs.len(),
            });
        }
        inputs.last_chunk().ok_or(OpError::InvalidKernelParameter)
    }

    pub(crate) fn validate(self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), OpError> {
        let bad = OpError::InvalidKernelParameter;
        let [state, table, generations, activity] = self.input_suffix(inputs)?;
        if [*table, *generations, *activity]
            .iter()
            .any(|t| t.layout().dtype() != DType::U32 || !t.layout().is_contiguous())
            || table.layout().shape() != [2, 2]
            || activity.layout().shape() != [1]
            || generations.layout().shape().len() != 1
            || self.access().source() >= generations.layout().shape()[0]
            || self.access().destination() >= generations.layout().shape()[0]
        {
            return Err(bad);
        }
        if !state.layout().is_contiguous()
            || !outputs[1].layout().is_contiguous()
            || state.layout().dtype() != outputs[1].layout().dtype()
            || state.layout().shape() != outputs[1].layout().shape()
            || inputs[..inputs.len() - METADATA_INPUTS]
                .iter()
                .chain(outputs.iter())
                .any(|t| !matches!(t.layout().dtype(), DType::F32 | DType::F16 | DType::BF16))
        {
            return Err(bad);
        }
        match self {
            Self::DeltaRule(_) => {
                let [heads, keys, values]: [u32; 3] =
                    state.layout().shape().try_into().map_err(|_| bad.clone())?;
                let q = inputs[0].layout().shape();
                if q.len() != 2
                    || q[1] != keys
                    || !heads.is_multiple_of(q[0])
                    || inputs[1].layout().shape() != q
                    || inputs[2].layout().shape() != [heads, values]
                    || inputs[3].layout().shape() != [heads]
                    || inputs[4].layout().shape() != [heads]
                    || outputs[0].layout().shape() != [heads, values]
                    || state.layout().dtype() != DType::F32
                {
                    return Err(bad);
                }
            }
            Self::DeltaRuleBlock(_) => super::block::validate(inputs, outputs)?,
            Self::CausalDepthwiseConv(_) => {
                let [tokens, channels]: [u32; 2] = inputs[0]
                    .layout()
                    .shape()
                    .try_into()
                    .map_err(|_| bad.clone())?;
                let weights = inputs[1].layout().shape();
                if weights.len() != 2
                    || weights[1] != channels
                    || weights[0] < 2
                    || state.layout().shape() != [weights[0] - 1, channels]
                    || outputs[0].layout().shape() != [tokens, channels]
                    || state.layout().dtype() != inputs[0].layout().dtype()
                    || outputs[0].layout().dtype() != inputs[0].layout().dtype()
                {
                    return Err(bad);
                }
            }
        }
        Ok(())
    }

    pub(crate) fn cost(self, inputs: &[Tensor], mut cost: KernelCost) -> Option<KernelCost> {
        cost.scalar_ops = AUTHENTICATION_SCALAR_OPS;
        cost.flops = match self {
            Self::DeltaRuleBlock(_) => {
                return crate::cost::state_block::stages(inputs, &inputs[2])
                    .and_then(|stages| crate::cost::state_block::composite(&stages));
            }
            Self::DeltaRule(_) => {
                let s = inputs[5].layout();
                cost.state_read_bytes = s.element_count().checked_mul(4)?;
                cost.state_write_bytes = cost.state_read_bytes;
                s.element_count().checked_mul(7)?.checked_add(
                    u64::from(s.shape()[0])
                        .checked_mul(u64::from(s.shape()[2]))?
                        .checked_mul(2)?,
                )?
            }
            Self::CausalDepthwiseConv(_) => return convolution_cost(inputs, cost),
        };
        Some(cost)
    }
}

fn convolution_cost(inputs: &[Tensor], mut cost: KernelCost) -> Option<KernelCost> {
    let tokens = u64::from(inputs[0].layout().shape()[0]);
    let channels = u64::from(inputs[0].layout().shape()[1]);
    let taps = u64::from(inputs[1].layout().shape()[0]);
    let history = taps.checked_sub(1)?;
    let covered = tokens.min(history);
    let triangle = if covered == 0 {
        0
    } else {
        covered
            .checked_mul(covered.checked_sub(1)?)?
            .checked_div(2)?
    };
    let history_convolution = covered.checked_mul(history)?.checked_sub(triangle)?;
    let history_carry = if history > tokens {
        history.checked_sub(tokens)?
    } else {
        0
    };
    cost.state_read_bytes = history_convolution
        .checked_add(history_carry)?
        .checked_mul(channels)?
        .checked_mul(inputs[2].layout().dtype().byte_size())?;
    cost.state_write_bytes = crate::cost::tensor_bytes(&inputs[2])?;
    let convolution = tokens.checked_mul(taps)?;
    let input_reads = convolution
        .checked_sub(history_convolution)?
        .checked_add(covered)?
        .checked_mul(channels)?
        .checked_mul(inputs[0].layout().dtype().byte_size())?;
    let weight_reads = convolution
        .checked_mul(channels)?
        .checked_mul(inputs[1].layout().dtype().byte_size())?;
    cost.read_bytes = input_reads
        .checked_add(weight_reads)?
        .checked_add(cost.state_read_bytes)?
        .checked_add(crate::cost::bytes(&inputs[3..])?)?;
    cost.flops = convolution.checked_mul(channels)?.checked_mul(2)?;
    Some(cost)
}

#[cfg(test)]
mod layout_tests {
    use super::*;
    #[test]
    fn authenticated_suffix_refuses_mismatched_operation_arity() {
        let pool = crate::StatePool::new(
            crate::StateLayout::new(vec![
                crate::StatePlane::new(DType::F32, vec![7, 33]).unwrap(),
            ])
            .unwrap(),
            1,
            65536,
        )
        .unwrap();
        let mut sequence = pool.reserve().unwrap();
        let transaction = sequence.begin(1).unwrap();
        let access = StateAccess::from_slot(&sequence.bind(&transaction).unwrap()).unwrap();
        let delta = StateOp::DeltaRule(access);
        assert!(delta.input_suffix(&[0; 6]).is_err());
        assert_eq!(
            delta.input_suffix(&[0, 1, 2, 3, 4, 5, 6, 7, 8]).unwrap(),
            &[5, 6, 7, 8]
        );
        let conv = StateOp::CausalDepthwiseConv(access);
        assert!(conv.input_suffix(&[0; 9]).is_err());
        assert_eq!(
            conv.input_suffix(&[0, 1, 2, 3, 4, 5]).unwrap(),
            &[2, 3, 4, 5]
        );
    }
}
