mod batch;

use crate::{Config, carry::Carry, numerics, quant::Affine};
use forja_sdk::{Result, Tensor, Weights, bf16, target::metal::Variant};

/// Gated delta-rule layer with opaque recurrent and raw convolution state.
pub struct DeltaNet {
    config: Config,
    qkv: Affine,
    z: Affine,
    ab: Affine,
    out: Affine,
    convolution: Tensor<bf16>,
    norm: Tensor<bf16>,
    norm_unit: Tensor<f32>,
    norm_unit_decode: Tensor<bf16>,
    log_a: Tensor<f32>,
    dt_bias: Tensor<f32>,
    carry: Carry,
    state_picks: [Option<Variant>; 2],
}
#[derive(Clone, Copy, Eq, PartialEq)]
enum Phase {
    Prefill,
    Decode,
}

struct DeltaProjection {
    query: Tensor<f32>,
    key: Tensor<f32>,
    value: Tensor<bf16>,
    beta: Tensor<bf16>,
    a: Tensor<bf16>,
}

impl DeltaNet {
    /// Loads exact projection shapes and allocates zeroed private state.
    ///
    /// # Errors
    /// Returns configuration, weight, quota or allocation errors.
    pub fn load(weights: &Weights<'_>, config: Config) -> Result<Self> {
        config.validate()?;
        Self::load_with_carry(weights, config, Carry::new(config)?)
    }

    /// Loads weights with a lane-local pair of host-owned serving planes.
    ///
    /// # Errors
    /// Refuses invalid planes, geometry or weights.
    pub fn load_paged(
        weights: &Weights<'_>,
        config: Config,
        cache: std::rc::Rc<forja_sdk::PagedCacheSet>,
        first: u32,
    ) -> Result<Self> {
        config.validate()?;
        Self::load_with_carry(weights, config, Carry::slotted(cache, first)?)
    }

    fn load_with_carry(weights: &Weights<'_>, config: Config, carry: Carry) -> Result<Self> {
        let channels = config.qkv()?;
        let values = config.value_heads * config.value_dim;
        let convolution = weights
            .tensor::<bf16>("conv1d.weight", &[channels, config.conv_width, 1])?
            .reshape(&[channels, config.conv_width])?
            .permute(&[1, 0])?
            .contiguous()?;
        let norm_unit = Tensor::constant(
            &vec![
                1.0_f32;
                usize::try_from(config.value_dim)
                    .map_err(|_| forja_sdk::Error::loading("head width exceeds address space"))?
            ],
            &[config.value_dim],
        )?;
        let norm_unit_decode = norm_unit.to_dtype()?;
        Ok(Self {
            config,
            qkv: Affine::load(&weights.scoped("in_proj_qkv"), config.hidden, channels)?,
            z: Affine::load(&weights.scoped("in_proj_z"), config.hidden, values)?,
            ab: Affine::load_pair(
                [&weights.scoped("in_proj_a"), &weights.scoped("in_proj_b")],
                config.hidden,
                config.value_heads,
            )?,
            out: Affine::load(&weights.scoped("out_proj"), values, config.hidden)?,
            convolution,
            norm: weights.tensor("norm.weight", &[config.value_dim])?,
            norm_unit,
            norm_unit_decode,
            log_a: weights
                .tensor::<bf16>("A_log", &[config.value_heads])?
                .to_dtype()?,
            dt_bias: weights
                .tensor::<bf16>("dt_bias", &[config.value_heads])?
                .to_dtype()?,
            carry,
            state_picks: [None, None],
        })
    }
    pub(crate) fn select_state_picks(&mut self, picks: [Option<Variant>; 2]) {
        self.state_picks = picks;
    }

    pub(crate) fn select_lane(&mut self, lane: u32) -> Result<()> {
        self.carry.select_lane(lane)
    }

    /// Starts a new sequence with zero convolution history and recurrent state.
    ///
    /// # Errors
    /// Returns host quota or allocation errors.
    pub fn reset(&mut self) -> Result<()> {
        if self.carry.is_slotted() {
            return Err(forja_sdk::Error::loading(
                "serving state reset belongs to the host",
            ));
        }
        self.carry = Carry::new(self.config)?;
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
        self.forward(input, pick, Phase::Decode, None)
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
        self.forward(input, quant, Phase::Prefill, delta)
    }
    fn forward(
        &mut self,
        input: &Tensor<bf16>,
        pick: Option<&Variant>,
        phase: Phase,
        delta: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        let prefill = phase == Phase::Prefill;
        let c = self.config;
        let tokens = input.shape().first().copied().unwrap_or(0);
        if input.shape() != [tokens, c.hidden]
            || tokens == 0
            || tokens
                > if self.carry.is_slotted() {
                    2048
                } else {
                    crate::PREFILL_BLOCK_TOKENS
                }
            || (!prefill && tokens != 1)
        {
            return Err(forja_sdk::Error::loading("invalid delta layer token span"));
        }
        let value_rows = rows(tokens, c.value_heads)?;
        let projected = self.qkv.forward(input, pick)?;
        let convolved = self.carry.convolve(
            &projected,
            &self.convolution,
            if prefill {
                None
            } else {
                self.state_picks[1].as_ref()
            },
        )?;
        let mixed = numerics::silu_bf16(&convolved)?;
        let gate_shape = if prefill {
            vec![tokens, c.value_heads]
        } else {
            vec![c.value_heads]
        };
        let DeltaProjection {
            query,
            key,
            value,
            beta,
            a,
        } = self.project_mix(&mixed, input, pick, &gate_shape)?;
        let y = if prefill {
            let bias = self.dt_bias.broadcast_as(&gate_shape)?;
            let log_a = self.log_a.broadcast_as(&gate_shape)?;
            let log_decay = numerics::log_decay(&a.to_dtype::<f32>()?, &bias, &log_a)?;
            self.carry
                .delta(
                    [
                        &query.reshape(&[tokens, c.key_heads, c.key_dim])?,
                        &key.reshape(&[tokens, c.key_heads, c.key_dim])?,
                    ],
                    &value.reshape(&[tokens, c.value_heads, c.value_dim])?,
                    &log_decay,
                    &beta,
                    true,
                    delta,
                )?
                .reshape(&[value_rows, c.value_dim])?
        } else {
            let decay = numerics::decay(&a, &self.dt_bias, &self.log_a)?;
            self.carry.delta(
                [&query, &key],
                &value,
                &decay,
                &beta,
                false,
                self.state_picks[0].as_ref(),
            )?
        };
        self.finish_rows(&y, input, pick, phase)
    }
    fn project_mix(
        &self,
        mixed: &Tensor<bf16>,
        input: &Tensor<bf16>,
        pick: Option<&Variant>,
        gate_shape: &[u32],
    ) -> Result<DeltaProjection> {
        let c = self.config;
        let tokens = input.shape()[0];
        let key_rows = rows(tokens, c.key_heads)?;
        let value_rows = rows(tokens, c.value_heads)?;
        let keys = c.key_heads * c.key_dim;
        let query =
            projected_panel(mixed, 0, keys, [key_rows, c.key_dim], tokens)?.to_dtype::<f32>()?;
        let key =
            projected_panel(mixed, keys, keys, [key_rows, c.key_dim], tokens)?.to_dtype::<f32>()?;
        let scale = numerics::inverse_sqrt_width(c.key_dim)?;
        let query = numerics::l2(&query, scale)?;
        let key = numerics::l2(&key, 1.0)?;
        let value = projected_panel(
            mixed,
            2 * keys,
            c.value_heads * c.value_dim,
            [value_rows, c.value_dim],
            tokens,
        )?;
        let ab = self.ab.forward(input, pick)?;
        let a = gate_panel(&ab, 0, c.value_heads, gate_shape)?;
        let beta =
            numerics::sigmoid_bf16(&gate_panel(&ab, c.value_heads, c.value_heads, gate_shape)?)?;
        Ok(DeltaProjection {
            query,
            key,
            value,
            beta,
            a,
        })
    }
    fn finish_rows(
        &self,
        y: &Tensor<bf16>,
        input: &Tensor<bf16>,
        pick: Option<&Variant>,
        phase: Phase,
    ) -> Result<Tensor<bf16>> {
        let c = self.config;
        let tokens = input.shape()[0];
        let value_rows = rows(tokens, c.value_heads)?;
        let z = self
            .z
            .forward(input, pick)?
            .reshape(&[value_rows, c.value_dim])?;
        let gated = match phase {
            Phase::Prefill => numerics::gated_norm(y, &self.norm, &z, &self.norm_unit)?,
            Phase::Decode => {
                numerics::gated_norm_decode(y, &self.norm, &z, &self.norm_unit_decode)?
            }
        }
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

fn projected_panel(
    mixed: &Tensor<bf16>,
    start: u32,
    width: u32,
    shape: [u32; 2],
    tokens: u32,
) -> Result<Tensor<bf16>> {
    let panel = mixed.narrow(1, start, width)?;
    if tokens == 1 {
        panel.reshape(&shape)
    } else {
        panel.contiguous()?.reshape(&shape)
    }
}

fn gate_panel(mixed: &Tensor<bf16>, start: u32, width: u32, shape: &[u32]) -> Result<Tensor<bf16>> {
    let panel = mixed.narrow(1, start, width)?;
    if shape.len() == 1 {
        panel.reshape(&[width])
    } else {
        Ok(panel)
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::{gate_panel, projected_panel};
    use forja_sdk::{NativeDevice, Tensor, bf16, set_native_device};

    #[test]
    fn projected_panel_uses_a_single_row_view_and_copies_multirow_panels() {
        set_native_device(NativeDevice::Cpu);
        for tokens in [1, 3] {
            let values = (0..tokens * 12)
                .map(|index| bf16::from_f64(f64::from(index)))
                .collect::<Vec<_>>();
            let mixed = Tensor::from_slice(&values, &[tokens, 12]).unwrap();
            let shape = [tokens * 2, 4];
            let actual = projected_panel(&mixed, 4, 8, shape, tokens).unwrap();
            let copied = mixed
                .narrow(1, 4, 8)
                .unwrap()
                .contiguous()
                .unwrap()
                .reshape(&shape)
                .unwrap();
            assert_eq!(actual.shape(), shape);
            assert_eq!(actual.to_vec().unwrap(), copied.to_vec().unwrap());

            let changed = (0..tokens * 12)
                .map(|index| bf16::from_f64(f64::from(index + 100)))
                .collect::<Vec<_>>();
            mixed.write(&changed).unwrap();
            if tokens == 1 {
                assert_eq!(actual.to_vec().unwrap(), changed[4..12]);
            } else {
                assert_eq!(actual.to_vec().unwrap(), copied.to_vec().unwrap());
            }
        }
    }

    #[test]
    fn gate_panel_preserves_batch_one_rank() {
        set_native_device(NativeDevice::Cpu);
        let mixed = Tensor::from_slice(&vec![bf16::from_f64(1.0); 96], &[1, 96]).unwrap();
        assert_eq!(gate_panel(&mixed, 48, 48, &[48]).unwrap().shape(), &[48]);
        assert_eq!(
            gate_panel(&mixed, 48, 48, &[1, 48]).unwrap().shape(),
            &[1, 48]
        );
    }

    #[test]
    fn gate_panels_match_reference_values_across_token_counts() {
        set_native_device(NativeDevice::Cpu);
        for tokens in [1, 3, 33] {
            let values = (0..tokens * 96)
                .map(|index| bf16::from_f64(f64::from(index % 251)))
                .collect::<Vec<_>>();
            let mixed = Tensor::from_slice(&values, &[tokens, 96]).unwrap();
            for start in [0, 48] {
                let actual = gate_panel(&mixed, start, 48, &[tokens, 48]).unwrap();
                let expected = values
                    .chunks_exact(96)
                    .flat_map(|row| row[start as usize..start as usize + 48].iter().copied())
                    .collect::<Vec<_>>();
                assert_eq!(actual.shape(), &[tokens, 48]);
                assert_eq!(actual.to_vec().unwrap(), expected);
            }
        }
    }
}
