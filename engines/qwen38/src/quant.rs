use forja_sdk::{Element, Result, Tensor, Weights, bf16, target::metal::Variant};

const TILED_MIN_ROWS: u32 = 16;

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
    /// Loads two equal-width projections as one output matrix.
    ///
    /// # Errors
    /// Refuses incompatible weights, failed reads or allocation errors.
    pub fn load_pair(weights: [&Weights<'_>; 2], input: u32, output: u32) -> Result<Self> {
        let first = Self::load(weights[0], input, output)?;
        let second = Self::load(weights[1], input, output)?;
        Self::pair(&first, &second)
    }

    fn pair(first: &Self, second: &Self) -> Result<Self> {
        Ok(Self {
            packed: concat_rows(&first.packed, &second.packed)?,
            scales: concat_rows(&first.scales, &second.scales)?,
            biases: concat_rows(&first.biases, &second.biases)?,
        })
    }
    /// Projects bf16 rows, accumulating in float32 before the output rounding.
    ///
    /// # Errors
    /// Refuses incompatible inputs or catalog selections.
    pub fn forward(&self, input: &Tensor<bf16>, pick: Option<&Variant>) -> Result<Tensor<bf16>> {
        match pick {
            Some(pick) => {
                let [rows, width]: [u32; 2] = input.shape().try_into().map_err(|_| {
                    forja_sdk::Error::loading("projection requires row-major input")
                })?;
                if matches!(pick.name(), "quant-matmul.tiled" | "quant-matmul.tiled-k64")
                    && (1..TILED_MIN_ROWS).contains(&rows)
                {
                    let padded = Tensor::zeros(&[TILED_MIN_ROWS, width])?;
                    input.copy_into(&mut padded.narrow(0, 0, rows)?)?;
                    padded
                        .quant_matmul_with(&self.packed, &self.scales, &self.biases, 4, 64, pick)?
                        .narrow(0, 0, rows)
                } else {
                    input.quant_matmul_with(&self.packed, &self.scales, &self.biases, 4, 64, pick)
                }
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

fn concat_rows<T: Element>(first: &Tensor<T>, second: &Tensor<T>) -> Result<Tensor<T>> {
    let [rows, width]: [u32; 2] = first
        .shape()
        .try_into()
        .map_err(|_| forja_sdk::Error::loading("paired projection requires matrices"))?;
    if second.shape() != [rows, width] {
        return Err(forja_sdk::Error::loading("paired projection shapes differ"));
    }
    let mut values = first.to_vec()?;
    values.extend(second.to_vec()?);
    Tensor::from_slice(
        &values,
        &[
            rows.checked_mul(2)
                .ok_or_else(|| forja_sdk::Error::loading("paired projection rows overflow"))?,
            width,
        ],
    )
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::Affine;
    use forja_sdk::{NativeDevice, Tensor, bf16, set_native_device};

    fn projection(word: u32, scale: f64, bias: f64) -> Affine {
        Affine {
            packed: Tensor::from_slice(&vec![word; 48 * 8], &[48, 8]).unwrap(),
            scales: Tensor::from_slice(&vec![bf16::from_f64(scale); 48], &[48, 1]).unwrap(),
            biases: Tensor::from_slice(&vec![bf16::from_f64(bias); 48], &[48, 1]).unwrap(),
        }
    }

    #[test]
    fn paired_affine_matches_individual_projections() {
        set_native_device(NativeDevice::Cpu);
        for rows in [1, 7] {
            let values = (0..rows * 64)
                .map(|i| bf16::from_f64(f64::from(i % 23) / 16.0 - 0.5))
                .collect::<Vec<_>>();
            let input = Tensor::from_slice(&values, &[rows, 64]).unwrap();
            let first = projection(0x7654_3210, 0.5, -0.125);
            let second = projection(0x89ab_cdef, 0.25, 0.375);
            let first_values = first.forward(&input, None).unwrap().to_vec().unwrap();
            let second_values = second.forward(&input, None).unwrap().to_vec().unwrap();
            let paired = Affine::pair(&first, &second).unwrap();
            let output = paired.forward(&input, None).unwrap();
            assert_eq!(
                output.narrow(1, 0, 48).unwrap().to_vec().unwrap(),
                first_values
            );
            assert_eq!(
                output.narrow(1, 48, 48).unwrap().to_vec().unwrap(),
                second_values
            );
        }
    }
}
