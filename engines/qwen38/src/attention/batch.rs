use super::{Attention, Cache};
use forja_sdk::{Dim, Error, Result, Tensor, bf16, f16, target::metal::Variant};

impl Attention {
    pub(crate) fn decode_rows(
        &self,
        input: &Tensor<bf16>,
        positions: &Tensor<u32>,
        starts: &[Dim],
        lanes: &[u32],
        quant: Option<&Variant>,
        attention: &Variant,
    ) -> Result<Tensor<bf16>> {
        let batch = u32::try_from(lanes.len())
            .map_err(|_| Error::loading("attention lane count exceeds u32"))?;
        if !(1..=8).contains(&batch)
            || input.shape() != [batch, self.config.hidden]
            || positions.shape() != [batch]
            || starts.len() != lanes.len()
            || lanes.iter().any(|lane| *lane >= 8)
            || lanes.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(Error::loading("invalid attention lane rows"));
        }
        let Cache::Paged { set, layer, .. } = &self.cache else {
            return Err(Error::loading("attention has no serving slot"));
        };
        let projected = self.project(input, positions, quant)?;
        let keys = projected.key.to_dtype::<f16>()?;
        let values = projected.value.to_dtype::<f16>()?;
        let width = self
            .config
            .query_heads
            .checked_mul(self.config.head_dim)
            .ok_or_else(|| Error::loading("attention row width overflow"))?;
        let attended = Tensor::<f32>::zeros(&[batch, width])?;
        let scale = crate::numerics::inverse_sqrt_width(self.config.head_dim)?;
        for (row, (&lane, start)) in lanes.iter().zip(starts).enumerate() {
            let row =
                u32::try_from(row).map_err(|_| Error::loading("attention lane row exceeds u32"))?;
            let cache = set.layer_at(lane, *layer)?;
            cache.append(&keys.narrow(0, row, 1)?, &values.narrow(0, row, 1)?, start)?;
            cache
                .attention(
                    &projected.query.narrow(0, row, 1)?,
                    scale,
                    start,
                    &start.clone().checked_add(1)?,
                    attention,
                )?
                .copy_into(&mut attended.narrow(0, row, 1)?)?;
        }
        self.output(&attended, &projected.gate, quant)
    }
}
