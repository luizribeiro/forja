use super::{DeltaNet, DeltaProjection, rows};
use crate::numerics;
use forja_sdk::{Error, Result, Tensor, bf16, target::metal::Variant};

impl DeltaNet {
    pub(crate) fn decode_rows(
        &mut self,
        input: &Tensor<bf16>,
        lanes: &[u32],
        pick: Option<&Variant>,
    ) -> Result<Tensor<bf16>> {
        let c = self.config;
        let batch =
            u32::try_from(lanes.len()).map_err(|_| Error::loading("lane count exceeds u32"))?;
        if input.shape() != [batch, c.hidden]
            || !(1..=8).contains(&batch)
            || lanes.iter().any(|lane| *lane >= 8)
            || lanes.windows(2).any(|pair| pair[0] >= pair[1])
            || !self.carry.is_slotted()
        {
            return Err(Error::loading("invalid recurrent lane rows"));
        }
        let projected = self.qkv.forward(input, pick)?;
        let convolved = Tensor::zeros(projected.shape())?;
        for (row, &lane) in lanes.iter().enumerate() {
            let row = u32::try_from(row).map_err(|_| Error::loading("lane row exceeds u32"))?;
            self.carry.select_lane(lane)?;
            self.carry.convolve_into(
                &projected.narrow(0, row, 1)?,
                &self.convolution,
                &mut convolved.narrow(0, row, 1)?,
            )?;
        }
        let mixed = numerics::silu_bf16(&convolved)?;
        let shape = [batch, c.value_heads];
        let DeltaProjection {
            query,
            key,
            value,
            beta,
            a,
        } = self.project_mix(&mixed, input, pick, &shape)?;
        let decay = numerics::decay(
            &a,
            &self.dt_bias.broadcast_as(&shape)?,
            &self.log_a.broadcast_as(&shape)?,
        )?;
        let y = Tensor::zeros(&[rows(batch, c.value_heads)?, c.value_dim])?;
        for (row, &lane) in lanes.iter().enumerate() {
            let row = u32::try_from(row).map_err(|_| Error::loading("lane row exceeds u32"))?;
            self.carry.select_lane(lane)?;
            self.carry.delta_into(
                [
                    &head_rows(&query, row, c.key_heads)?,
                    &head_rows(&key, row, c.key_heads)?,
                ],
                &head_rows(&value, row, c.value_heads)?,
                &decay.narrow(0, row, 1)?.reshape(&[c.value_heads])?,
                &beta.narrow(0, row, 1)?.reshape(&[c.value_heads])?,
                &mut head_rows(&y, row, c.value_heads)?,
            )?;
        }
        self.finish_rows(&y, input, pick)
    }
}
fn head_rows<T: forja_sdk::Element>(tensor: &Tensor<T>, row: u32, heads: u32) -> Result<Tensor<T>> {
    let start = row
        .checked_mul(heads)
        .ok_or_else(|| Error::loading("lane row offset overflow"))?;
    tensor.narrow(0, start, heads)
}
