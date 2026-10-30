use crate::Config;
use forja_sdk::{DType, FloatElement, Result, Tensor, state::StatePlane, target::metal::Variant};

pub(super) struct Carry {
    history: StatePlane,
    recurrent: StatePlane,
}

impl Carry {
    pub(super) fn new(config: Config) -> Result<Self> {
        Ok(Self {
            history: StatePlane::new(DType::BF16, &[config.conv_width - 1, config.qkv()?])?,
            recurrent: StatePlane::new(
                DType::F32,
                &[config.value_heads, config.key_dim, config.value_dim],
            )?,
        })
    }

    pub(super) fn convolve<T: FloatElement>(
        &mut self,
        input: &Tensor<T>,
        weights: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        self.history.causal_convolution(input, weights)
    }

    pub(super) fn delta<T: FloatElement>(
        &mut self,
        qk: [&Tensor<f32>; 2],
        value: &Tensor<T>,
        decay: &Tensor<f32>,
        beta: &Tensor<T>,
        block: bool,
        pick: Option<&Variant>,
    ) -> Result<Tensor<T>> {
        if block {
            self.recurrent
                .delta_rule_block(qk[0], qk[1], value, decay, beta, pick)
        } else {
            self.recurrent.delta_rule(qk[0], qk[1], value, decay, beta)
        }
    }
}
