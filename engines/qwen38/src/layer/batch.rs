use super::{DecoderLayer, Mixer};
use forja_sdk::{Dim, Result, Tensor, bf16, target::metal::Variant};

impl DecoderLayer {
    pub(crate) fn decode_rows(
        &mut self,
        input: &Tensor<bf16>,
        positions: &Tensor<u32>,
        starts: &[Dim],
        lanes: &[u32],
        quant: Option<&Variant>,
        attention: &Variant,
    ) -> Result<Tensor<bf16>> {
        let normalized = self.normalize(input)?;
        let mixed = match &mut self.mixer {
            Mixer::Recurrent(layer) => layer.decode_rows(&normalized, lanes, quant)?,
            Mixer::Attention(layer) => {
                layer.decode_rows(&normalized, positions, starts, lanes, quant, attention)?
            }
        };
        self.finish(input, &mixed, quant)
    }
}
