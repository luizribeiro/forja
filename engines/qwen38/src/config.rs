use forja_sdk::{Error, Result};

/// Validated dimensions shared by the production model and random-weight tests.
#[derive(Clone, Copy, Debug)]
pub struct Config {
    /// Vocabulary size.
    pub vocab: u32,
    /// Residual width.
    pub hidden: u32,
    /// Feed-forward intermediate width.
    pub intermediate: u32,
    /// Number of decoder layers.
    pub layers: u32,
    /// Full attention occurs at every `attention_interval` layers.
    pub attention_interval: u32,
    /// Query heads in full attention.
    pub query_heads: u32,
    /// Key/value heads in full attention.
    pub kv_heads: u32,
    /// Full attention head width.
    pub head_dim: u32,
    /// Rotated prefix of each full attention head.
    pub rotary_dim: u32,
    /// Recurrent key heads.
    pub key_heads: u32,
    /// Recurrent value heads.
    pub value_heads: u32,
    /// Recurrent key width.
    pub key_dim: u32,
    /// Recurrent value width.
    pub value_dim: u32,
    /// Raw convolution window.
    pub conv_width: u32,
    /// Native CPU tests use portable operations.
    pub cpu: bool,
}
impl Config {
    /// Dimensions of the pinned checkpoint.
    #[must_use]
    pub const fn model() -> Self {
        Self {
            vocab: 248_320,
            hidden: 5120,
            intermediate: 17_408,
            layers: 64,
            attention_interval: 4,
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rotary_dim: 64,
            key_heads: 16,
            value_heads: 48,
            key_dim: 128,
            value_dim: 128,
            conv_width: 4,
            cpu: false,
        }
    }
    /// Small four-layer model retaining both attention mechanisms and grouped heads.
    #[must_use]
    pub const fn tiny() -> Self {
        Self {
            vocab: 33,
            hidden: 64,
            intermediate: 128,
            layers: 4,
            attention_interval: 4,
            query_heads: 2,
            kv_heads: 1,
            head_dim: 32,
            rotary_dim: 8,
            key_heads: 1,
            value_heads: 4,
            key_dim: 16,
            value_dim: 16,
            conv_width: 4,
            cpu: true,
        }
    }
    /// Describes convolution history and float32 carry for every recurrent layer.
    ///
    /// # Errors
    /// Refuses invalid configuration or overflowing projection widths.
    pub fn state_planes(self) -> Result<Vec<forja_sdk::PagedStateDescriptor>> {
        self.validate()?;
        let channels = self.qkv()?;
        Ok((0..self.layers)
            .filter(|&layer| !self.full_attention(layer))
            .flat_map(|_| {
                [
                    forja_sdk::PagedStateDescriptor {
                        dtype: forja_sdk::DType::BF16,
                        shape: vec![self.conv_width - 1, channels],
                    },
                    forja_sdk::PagedStateDescriptor {
                        dtype: forja_sdk::DType::F32,
                        shape: vec![self.value_heads, self.key_dim, self.value_dim],
                    },
                ]
            })
            .collect())
    }

    /// Checks projection packing, attention grouping and bounded dimensions.
    ///
    /// # Errors
    /// Refuses invalid dimensions or overflowing projection widths.
    pub fn validate(self) -> Result<()> {
        let dimensions = [
            self.vocab,
            self.hidden,
            self.intermediate,
            self.layers,
            self.attention_interval,
            self.query_heads,
            self.kv_heads,
            self.head_dim,
            self.rotary_dim,
            self.key_heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
            self.conv_width,
        ];
        if dimensions.contains(&0)
            || self.conv_width < 2
            || self.rotary_dim > self.head_dim
            || !self.rotary_dim.is_multiple_of(2)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || !self.value_heads.is_multiple_of(self.key_heads)
            || self.key_dim > 256
        {
            return Err(Error::loading("invalid hybrid attention geometry"));
        }
        let products = [
            (self.key_heads, self.key_dim),
            (self.value_heads, self.value_dim),
            (self.query_heads, self.head_dim),
            (self.kv_heads, self.head_dim),
        ];
        let widths = products.map(|(a, b)| a.checked_mul(b));
        let [Some(keys), Some(values), Some(queries), Some(_)] = widths else {
            return Err(Error::loading("model width overflow"));
        };
        if keys
            .checked_mul(2)
            .and_then(|k| k.checked_add(values))
            .is_none()
            || queries.checked_mul(2).is_none()
            || [self.hidden, self.intermediate, values, queries]
                .iter()
                .any(|n| !n.is_multiple_of(64))
        {
            return Err(Error::loading("invalid affine-quantized projection width"));
        }
        Ok(())
    }
    /// Combined recurrent Q/K/V projection width.
    ///
    /// # Errors
    /// Refuses arithmetic overflow.
    pub fn qkv(self) -> Result<u32> {
        self.key_heads
            .checked_mul(self.key_dim)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| self.value_heads.checked_mul(self.value_dim)?.checked_add(n))
            .ok_or_else(|| Error::loading("recurrent projection width overflow"))
    }
    /// Whether a decoder layer uses full attention.
    #[must_use]
    pub fn full_attention(self, layer: u32) -> bool {
        layer
            .checked_add(1)
            .is_some_and(|n| n.checked_rem(self.attention_interval) == Some(0))
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_both_model_geometries_and_refuses_overflow() {
        for config in [Config::model(), Config::tiny()] {
            config.validate().unwrap();
            assert!(!config.full_attention(0));
            assert!(config.full_attention(3));
        }
        let mut config = Config::tiny();
        config.key_dim = u32::MAX;
        assert!(config.validate().is_err());
        config = Config::tiny();
        config.kv_heads = 3;
        assert!(config.validate().is_err());
        config = Config::tiny();
        config.hidden = 7;
        assert!(config.validate().is_err());
    }
}
