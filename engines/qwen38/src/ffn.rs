use crate::{Config, numerics, quant::Affine};
use forja_sdk::{Result, Tensor, Weights, bf16, target::metal::Variant};
/// Dense `SwiGLU` feed-forward block.
pub struct FeedForward {
    gate: Affine,
    up: Affine,
    down: Affine,
}
impl FeedForward {
    /// Loads the exact dense feed-forward projection shapes.
    ///
    /// # Errors
    /// Returns invalid configuration or weight metadata errors.
    pub fn load(weights: &Weights<'_>, config: Config) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            gate: Affine::load(
                &weights.scoped("gate_proj"),
                config.hidden,
                config.intermediate,
            )?,
            up: Affine::load(
                &weights.scoped("up_proj"),
                config.hidden,
                config.intermediate,
            )?,
            down: Affine::load(
                &weights.scoped("down_proj"),
                config.intermediate,
                config.hidden,
            )?,
        })
    }
    /// Applies `SiLU` with its bf16 rounding boundary before multiplying the up projection.
    ///
    /// # Errors
    /// Returns incompatible input, variant or execution errors.
    pub fn ffn(&self, input: &Tensor<bf16>, pick: Option<&Variant>) -> Result<Tensor<bf16>> {
        let gate = numerics::silu_bf16(&self.gate.forward(input, pick)?)?;
        let up = self.up.forward(input, pick)?;
        self.down.forward(&numerics::multiply(&gate, &up)?, pick)
    }
}
