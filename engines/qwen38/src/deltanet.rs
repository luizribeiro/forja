use crate::{Config, numerics, quant::Affine};
use forja_sdk::{DType, Result, Tensor, Weights, bf16, state::StatePlane, target::metal::Variant};

/// Gated delta-rule layer with opaque recurrent and raw convolution state.
pub struct DeltaNet {
    config: Config,
    qkv: Affine,
    z: Affine,
    a: Affine,
    b: Affine,
    out: Affine,
    convolution: Tensor<bf16>,
    norm: Tensor<bf16>,
    log_a: Tensor<f32>,
    dt_bias: Tensor<f32>,
    history: StatePlane,
    recurrent: StatePlane,
}
impl DeltaNet {
    /// Loads exact projection shapes and allocates zeroed private state.
    ///
    /// # Errors
    /// Returns configuration, weight, quota or allocation errors.
    pub fn load(weights: &Weights<'_>, config: Config) -> Result<Self> {
        config.validate()?;
        let channels = config.qkv()?;
        let values = config.value_heads * config.value_dim;
        let convolution = weights
            .tensor::<bf16>("conv1d.weight", &[channels, config.conv_width, 1])?
            .reshape(&[channels, config.conv_width])?
            .permute(&[1, 0])?
            .contiguous()?;
        Ok(Self {
            config,
            qkv: Affine::load(&weights.scoped("in_proj_qkv"), config.hidden, channels)?,
            z: Affine::load(&weights.scoped("in_proj_z"), config.hidden, values)?,
            a: Affine::load(
                &weights.scoped("in_proj_a"),
                config.hidden,
                config.value_heads,
            )?,
            b: Affine::load(
                &weights.scoped("in_proj_b"),
                config.hidden,
                config.value_heads,
            )?,
            out: Affine::load(&weights.scoped("out_proj"), values, config.hidden)?,
            convolution,
            norm: weights.tensor("norm.weight", &[config.value_dim])?,
            log_a: weights
                .tensor::<bf16>("A_log", &[config.value_heads])?
                .to_dtype()?,
            dt_bias: weights
                .tensor::<bf16>("dt_bias", &[config.value_heads])?
                .to_dtype()?,
            history: StatePlane::new(DType::BF16, &[config.conv_width - 1, channels])?,
            recurrent: StatePlane::new(
                DType::F32,
                &[config.value_heads, config.key_dim, config.value_dim],
            )?,
        })
    }
    /// Starts a new sequence with zero convolution history and recurrent state.
    ///
    /// # Errors
    /// Returns host quota or allocation errors.
    pub fn reset(&mut self) -> Result<()> {
        let c = self.config;
        self.history = StatePlane::new(DType::BF16, &[c.conv_width - 1, c.qkv()?])?;
        self.recurrent = StatePlane::new(DType::F32, &[c.value_heads, c.key_dim, c.value_dim])?;
        Ok(())
    }
    /// Processes one normalized token, publishing each private plane after completion.
    ///
    /// # Errors
    /// Refuses non-single-token inputs or failed projection/state operations.
    pub fn deltanet_layer(
        &mut self,
        input: &Tensor<bf16>,
        pick: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        self.forward(input, pick, false, None)
    }
    /// Applies a canonical prefill block, including its masked final tail.
    ///
    /// # Errors
    /// Refuses empty or oversized blocks, invalid picks or failed state operations.
    pub fn prefill(
        &mut self,
        input: &Tensor<bf16>,
        quant: Option<&Variant>,
        delta: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        self.forward(input, quant, true, delta)
    }
    fn forward(
        &mut self,
        input: &Tensor<bf16>,
        pick: Option<&Variant>,
        prefill: bool,
        delta: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        let c = self.config;
        let tokens = input.shape().first().copied().unwrap_or(0);
        if input.shape() != [tokens, c.hidden]
            || tokens == 0
            || tokens > crate::PREFILL_BLOCK_TOKENS
            || (!prefill && tokens != 1)
        {
            return Err(forja_sdk::Error::loading("invalid delta layer token span"));
        }
        let key_rows = rows(tokens, c.key_heads)?;
        let value_rows = rows(tokens, c.value_heads)?;
        let projected = self.qkv.forward(input, pick)?;
        let convolved = self
            .history
            .causal_convolution(&projected, &self.convolution)?;
        let mixed = numerics::silu_bf16(&convolved)?;
        let keys = c.key_heads * c.key_dim;
        let query = mixed
            .narrow(1, 0, keys)?
            .contiguous()?
            .reshape(&[key_rows, c.key_dim])?
            .to_dtype::<f32>()?;
        let key = mixed
            .narrow(1, keys, keys)?
            .contiguous()?
            .reshape(&[key_rows, c.key_dim])?
            .to_dtype::<f32>()?;
        let scale = numerics::inverse_sqrt_width(c.key_dim)?;
        let query = numerics::l2(&query, scale)?;
        let key = numerics::l2(&key, 1.0)?;
        let value = mixed
            .narrow(1, 2 * keys, c.value_heads * c.value_dim)?
            .contiguous()?
            .reshape(&[value_rows, c.value_dim])?;
        let gate_shape = if prefill {
            vec![tokens, c.value_heads]
        } else {
            vec![c.value_heads]
        };
        let beta = numerics::sigmoid_bf16(&self.b.forward(input, pick)?)?.reshape(&gate_shape)?;
        let a = self
            .a
            .forward(input, pick)?
            .reshape(&gate_shape)?
            .to_dtype::<f32>()?;
        let y = if prefill {
            let bias = self.dt_bias.broadcast_as(&gate_shape)?;
            let log_a = self.log_a.broadcast_as(&gate_shape)?;
            let log_decay = numerics::log_decay(&a, &bias, &log_a)?;
            self.recurrent
                .delta_rule_block(
                    &query.reshape(&[tokens, c.key_heads, c.key_dim])?,
                    &key.reshape(&[tokens, c.key_heads, c.key_dim])?,
                    &value.reshape(&[tokens, c.value_heads, c.value_dim])?,
                    &log_decay,
                    &beta,
                    delta,
                )?
                .reshape(&[value_rows, c.value_dim])?
        } else {
            let decay = numerics::decay(&a, &self.dt_bias, &self.log_a)?;
            self.recurrent
                .delta_rule(&query, &key, &value, &decay, &beta)?
        };
        let z = self
            .z
            .forward(input, pick)?
            .reshape(&[value_rows, c.value_dim])?;
        let gated = numerics::gated_norm(&y, &self.norm, &z)?
            .reshape(&[tokens, c.value_heads * c.value_dim])?;
        self.out.forward(&gated, pick)
    }
}

fn rows(tokens: u32, heads: u32) -> Result<u32> {
    u64::from(tokens)
        .checked_mul(u64::from(heads))
        .and_then(|count| u32::try_from(count).ok())
        .ok_or_else(|| forja_sdk::Error::loading("delta panel row count overflow"))
}
