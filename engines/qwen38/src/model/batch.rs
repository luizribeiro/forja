use super::Model;
use forja_sdk::{Dim, Error, Result, Tensor};

impl Model {
    pub(crate) fn forward_decode_rows(
        &mut self,
        tokens: &Tensor<u32>,
        starts: &[Dim],
    ) -> Result<Tensor<f32>> {
        let batch = u32::try_from(starts.len())
            .map_err(|_| Error::loading("decode row count exceeds u32"))?;
        if ![1, 2, 4, 8].contains(&batch)
            || tokens.shape() != [batch]
            || starts.iter().any(|start| start.value() >= self.context)
        {
            return Err(Error::loading("invalid decode row span"));
        }
        let paged = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::loading("model has no serving slot"))?;
        let positions = Tensor::<u32>::zeros(&[batch])?;
        for (row, start) in starts.iter().enumerate() {
            let row =
                u32::try_from(row).map_err(|_| Error::loading("decode row index exceeds u32"))?;
            paged
                .positions
                .narrow(0, start.clone(), 1)?
                .copy_into(&mut positions.narrow(0, row, 1)?)?;
        }
        let lanes = (0..batch).collect::<Vec<_>>();
        forja_sdk::profile_scope("embedding", "global", None)?;
        let mut hidden = self.embedding.embed(tokens)?;
        for layer in &mut self.layers {
            hidden = layer.decode_rows(
                &hidden,
                &positions,
                starts,
                &lanes,
                self.quant.as_ref(),
                &paged.decode,
            )?;
        }
        forja_sdk::profile_scope("output", "global", None)?;
        hidden = hidden.rms_norm(&self.norm, 1.0e-6)?;
        self.output
            .forward(&hidden, self.quant.as_ref())?
            .to_dtype::<f32>()
    }
}
