//! Configuration for the `SmolLM2` Llama engine.

#![forbid(unsafe_code)]

/// Dimensions shared by the reference and optimized implementations.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    vocab: u32,
    hidden: u32,
    intermediate: u32,
    query_heads: u32,
    kv_heads: u32,
    head_dim: u32,
    layers: usize,
    context: u32,
    epsilon: f32,
    theta: f32,
}

impl Config {
    /// Pinned Hugging Face model architecture.
    #[must_use]
    pub const fn smollm2() -> Self {
        Self {
            vocab: 49_152,
            hidden: 576,
            intermediate: 1_536,
            query_heads: 9,
            kv_heads: 3,
            head_dim: 64,
            layers: 30,
            context: 8_192,
            epsilon: 1.0e-5,
            theta: 100_000.0,
        }
    }

    /// Small instance of the same GQA, rotary, tied-embedding architecture.
    #[must_use]
    pub const fn tiny() -> Self {
        Self {
            vocab: 33,
            hidden: 24,
            intermediate: 40,
            query_heads: 3,
            kv_heads: 1,
            head_dim: 8,
            layers: 2,
            context: 64,
            epsilon: 1.0e-5,
            theta: 100_000.0,
        }
    }

    /// Inverse square root of the attention head width.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn attention_scale(self) -> f32 {
        (self.head_dim as f32).sqrt().recip()
    }

    /// Checks architecture arithmetic before loading or allocating tensors.
    ///
    /// # Errors
    /// Returns an error for incompatible or overflowing model dimensions.
    pub fn validate(self) -> forja_sdk::Result<()> {
        let heads = self.query_heads.checked_mul(self.head_dim);
        let kv = self.kv_heads.checked_mul(self.head_dim);
        if self.vocab == 0
            || self.hidden == 0
            || self.intermediate == 0
            || self.layers == 0
            || self.context == 0
            || self.kv_heads == 0
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || heads != Some(self.hidden)
            || kv.is_none()
            || self.head_dim == 0
            || !self.head_dim.is_multiple_of(2)
            || !self.epsilon.is_finite()
            || self.epsilon <= 0.0
            || !self.theta.is_finite()
            || self.theta <= 0.0
        {
            return Err(forja_sdk::Error::loading("invalid Llama dimensions"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn real_and_tiny_configs_use_grouped_rotary_attention() {
        for config in [Config::smollm2(), Config::tiny()] {
            config.validate().unwrap();
            assert_eq!(config.query_heads / config.kv_heads, 3);
            assert_eq!(config.hidden, config.query_heads * config.head_dim);
        }
        let mut invalid = Config::tiny();
        invalid.kv_heads = 2;
        assert!(invalid.validate().is_err());
        invalid.kv_heads = 0;
        assert!(invalid.validate().is_err());
    }
}
