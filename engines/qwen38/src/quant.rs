use forja_sdk::{Result, Tensor, Weights, bf16, target::metal::Variant};

/// Four-bit affine weights with one scale and bias per 64 columns.
pub struct Affine {
    packed: Tensor<u32>,
    scales: Tensor<bf16>,
    biases: Tensor<bf16>,
}
impl Affine {
    /// Validates and loads a projection's packed words, scales and biases.
    ///
    /// # Errors
    /// Refuses unsupported geometry or missing/mismatched tensors.
    pub fn load(weights: &Weights<'_>, input: u32, output: u32) -> Result<Self> {
        if input == 0 || output == 0 || !input.is_multiple_of(64) {
            return Err(forja_sdk::Error::loading(
                "invalid affine projection geometry",
            ));
        }
        Ok(Self {
            packed: weights.tensor("weight", &[output, input / 8])?,
            scales: weights.tensor("scales", &[output, input / 64])?,
            biases: weights.tensor("biases", &[output, input / 64])?,
        })
    }
    /// Projects bf16 rows, accumulating in float32 before the output rounding.
    ///
    /// # Errors
    /// Refuses incompatible inputs or catalog selections.
    pub fn forward(&self, input: &Tensor<bf16>, pick: Option<&Variant>) -> Result<Tensor<bf16>> {
        match pick {
            Some(pick) => {
                input.quant_matmul_with(&self.packed, &self.scales, &self.biases, 4, 64, pick)
            }
            None => input.quant_matmul(&self.packed, &self.scales, &self.biases, 4, 64),
        }
    }
    /// Dequantizes selected embedding rows.
    ///
    /// # Errors
    /// Refuses out-of-range token ids or failed dispatches.
    pub fn embed(&self, tokens: &Tensor<u32>) -> Result<Tensor<bf16>> {
        tokens.quant_embedding(&self.packed, &self.scales, &self.biases, 4, 64)
    }
}
