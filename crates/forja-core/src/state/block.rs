use crate::{DType, OpError, Tensor};

/// Checked float32 scratch geometry for a fixed-size delta-rule block transform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeltaBlockScratch {
    blocks: u32,
    stride: u32,
    elements: u32,
    offsets: [u32; 8],
}

impl DeltaBlockScratch {
    /// Tokens in one numerical block, independent of submission grouping.
    pub const TOKENS: usize = 64;

    /// Plans Gram matrices, decay factors, solved panels and carry/output scratch.
    ///
    /// # Errors
    /// Refuses zero dimensions, arithmetic overflow or a scratch extent above u32.
    pub fn new(tokens: u32, heads: u32, keys: u32, values: u32) -> Result<Self, OpError> {
        let bad = || OpError::InvalidKernelParameter;
        if [tokens, heads, keys, values].contains(&0) {
            return Err(bad());
        }
        let block = u64::try_from(Self::TOKENS).map_err(|_| bad())?;
        let matrix = block.checked_mul(block).ok_or_else(bad)?;
        let key_panel = block.checked_mul(u64::from(keys)).ok_or_else(bad)?;
        let value_panel = block.checked_mul(u64::from(values)).ok_or_else(bad)?;
        let sizes = [
            matrix,
            matrix,
            block,
            block,
            value_panel,
            key_panel,
            value_panel,
            value_panel,
        ];
        let mut stride = 0_u64;
        let mut offsets = [0_u32; 8];
        for (offset, size) in offsets.iter_mut().zip(sizes) {
            *offset = u32::try_from(stride).map_err(|_| bad())?;
            stride = stride.checked_add(size).ok_or_else(bad)?;
        }
        let blocks = u64::from(tokens).div_ceil(block);
        let elements = stride
            .checked_mul(blocks)
            .and_then(|n| n.checked_mul(u64::from(heads)))
            .ok_or_else(bad)?;
        Ok(Self {
            blocks: u32::try_from(blocks).map_err(|_| bad())?,
            stride: u32::try_from(stride).map_err(|_| bad())?,
            elements: u32::try_from(elements).map_err(|_| bad())?,
            offsets,
        })
    }

    /// Independent numerical blocks in each head's transform.
    #[must_use]
    pub const fn blocks(self) -> u32 {
        self.blocks
    }

    /// Float32 elements retained per head and numerical block.
    #[must_use]
    pub const fn stride(self) -> u32 {
        self.stride
    }

    /// Flat float32 allocation extent, including the masked final block.
    #[must_use]
    pub const fn elements(self) -> u32 {
        self.elements
    }

    /// Offsets of lower Gram, query Gram, prefix/tail decay, U/W, delta and base output.
    #[must_use]
    pub const fn offsets(self) -> [u32; 8] {
        self.offsets
    }
}

pub(super) fn validate(inputs: &[&Tensor], outputs: &[&Tensor]) -> Result<(), OpError> {
    let state = inputs[5];
    let bad = OpError::InvalidKernelParameter;
    let [heads, keys, values]: [u32; 3] =
        state.layout().shape().try_into().map_err(|_| bad.clone())?;
    let query = inputs[0].layout().shape();
    if query.len() != 3 || query[1] == 0 || !heads.is_multiple_of(query[1]) || query[2] != keys {
        return Err(bad);
    }
    let tokens = query[0];
    let scratch = DeltaBlockScratch::new(tokens, heads, keys, values)?;
    if inputs[1].layout().shape() != query
        || inputs[2].layout().shape() != [tokens, heads, values]
        || inputs[3].layout().shape() != [tokens, heads]
        || inputs[4].layout().shape() != [tokens, heads]
        || outputs[0].layout().shape() != inputs[2].layout().shape()
        || inputs[..2]
            .iter()
            .chain([&inputs[3]])
            .any(|t| t.layout().dtype() != DType::F32)
        || state.layout().dtype() != DType::F32
        || outputs[2].layout().dtype() != DType::F32
        || !outputs[2].layout().is_contiguous()
        || outputs[2].layout().shape() != [scratch.elements()]
        || inputs[4].layout().dtype() != inputs[2].layout().dtype()
        || outputs[0].layout().dtype() != inputs[2].layout().dtype()
    {
        return Err(bad);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panels_have_checked_disjoint_extents_including_a_partial_block() {
        let layout = DeltaBlockScratch::new(65, 3, 7, 33).unwrap();
        assert_eq!(layout.blocks(), 2);
        assert_eq!(layout.stride(), 15104);
        assert_eq!(layout.elements(), 90624);
        assert_eq!(
            layout.offsets(),
            [0, 4096, 8192, 8256, 8320, 10432, 10880, 12992]
        );
        for (tokens, blocks) in [(1, 1), (63, 1), (64, 1), (65, 2), (4097, 65)] {
            assert_eq!(
                DeltaBlockScratch::new(tokens, 6, 128, 33).unwrap().blocks(),
                blocks
            );
        }
    }

    #[test]
    fn scratch_geometry_refuses_zero_and_overflow() {
        for shape in [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
            [u32::MAX, u32::MAX, u32::MAX, u32::MAX],
            [1, 1, 1, u32::MAX],
        ] {
            assert!(DeltaBlockScratch::new(shape[0], shape[1], shape[2], shape[3]).is_err());
        }
    }
}
