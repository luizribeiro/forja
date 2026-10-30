use crate::{DType, KernelCost, OpError, StateAccess, Tensor};

// Activity, two expected identities, disjointness, two bounds and two generations.
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

    pub(crate) fn validate(self, inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), OpError> {
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
        let bad = OpError::InvalidKernelParameter;
        let [table, generations, activity] = &inputs[inputs.len() - 3..] else {
            return Err(bad);
        };
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
        let state = inputs[inputs.len() - 4];
        if !state.layout().is_contiguous()
            || !outputs[1].layout().is_contiguous()
            || state.layout().dtype() != outputs[1].layout().dtype()
            || state.layout().shape() != outputs[1].layout().shape()
            || inputs[..inputs.len() - 3]
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
            Self::DeltaRuleBlock(_) => return None,
            Self::DeltaRule(_) => {
                let s = inputs[5].layout();
                s.element_count().checked_mul(7)?.checked_add(
                    u64::from(s.shape()[0])
                        .checked_mul(u64::from(s.shape()[2]))?
                        .checked_mul(2)?,
                )?
            }
            Self::CausalDepthwiseConv(_) => inputs[0]
                .layout()
                .element_count()
                .checked_mul(u64::from(inputs[1].layout().shape()[0]))?
                .checked_mul(2)?,
        };
        Some(cost)
    }
}
