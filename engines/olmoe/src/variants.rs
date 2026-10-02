use forja_sdk::{
    DType, EngineLoadConfig, EngineSelections, Param, Result, VariantSelection,
    target::metal::Variant,
};

const SITES: &[&str] = &[
    "dense.decode",
    "dense.decode-transposed",
    "dense.prefill-bf16-large-wide",
    "dense.prefill-bf16-large",
    "dense.prefill-bf16-small",
    "dense.prefill-bf16-transposed",
    "dense.prefill-f32-large",
    "dense.prefill-f32-small",
    "dense.prefill-f32-transposed",
    "top-k",
    "attention.decode",
    "attention.prefill-large",
    "attention.prefill-small",
];

pub(crate) struct Variants(EngineSelections);

impl Variants {
    pub(crate) fn new(config: &EngineLoadConfig) -> Result<Self> {
        Ok(Self(EngineSelections::new(
            config,
            &[],
            SITES,
            crate::MAX_CONTEXT,
        )?))
    }

    pub(crate) fn matmul(
        &self,
        dtype: DType,
        batch: u32,
        rows: u32,
        columns: u32,
        inner: u32,
        transposed_right: bool,
    ) -> Result<Variant> {
        self.0.at(
            matmul_site(dtype, batch, rows, columns, inner, transposed_right)?,
            0,
        )
    }

    pub(crate) fn top_k(&self) -> Result<Variant> {
        self.0.at("top-k", 0)
    }

    pub(crate) fn attention(
        &self,
        sequence: u32,
        start: u32,
        parameter: Option<&Param>,
    ) -> Result<VariantSelection> {
        let site = if sequence == 1 {
            "attention.decode"
        } else if sequence >= 512 {
            "attention.prefill-large"
        } else {
            "attention.prefill-small"
        };
        if let Some(parameter) = parameter {
            self.0.for_param(site, parameter)
        } else {
            Ok(VariantSelection::Fixed(self.0.at(site, start)?))
        }
    }
}

fn matmul_site(
    dtype: DType,
    batch: u32,
    rows: u32,
    columns: u32,
    inner: u32,
    transposed_right: bool,
) -> Result<&'static str> {
    if rows == 1 {
        return Ok(if transposed_right {
            "dense.decode-transposed"
        } else {
            "dense.decode"
        });
    }
    let large = u64::from(batch)
        .checked_mul(u64::from(rows))
        .and_then(|elements| elements.checked_mul(u64::from(columns)))
        .ok_or_else(|| forja_sdk::Error::loading("matmul output size overflowed"))?
        >= 1 << 20;
    let half = matches!(dtype, DType::F16 | DType::BF16);
    Ok(
        if half && large && u64::from(rows.max(columns)) * 2 > u64::from(inner) {
            "dense.prefill-bf16-large-wide"
        } else if half && transposed_right {
            "dense.prefill-bf16-transposed"
        } else if half && large {
            "dense.prefill-bf16-large"
        } else if half {
            "dense.prefill-bf16-small"
        } else if !large && transposed_right {
            "dense.prefill-f32-transposed"
        } else if !large {
            "dense.prefill-f32-small"
        } else {
            "dense.prefill-f32-large"
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_dense_dispatch_sites_at_boundaries() {
        assert_eq!(
            matmul_site(DType::F32, 1, 1, crate::HIDDEN, crate::HIDDEN, true).unwrap(),
            "dense.decode-transposed"
        );
        assert_eq!(
            matmul_site(DType::F32, 1, 511, crate::HIDDEN, crate::HIDDEN, true).unwrap(),
            "dense.prefill-f32-transposed"
        );
        assert_eq!(
            matmul_site(DType::F32, 1, 512, crate::HIDDEN, crate::HIDDEN, true).unwrap(),
            "dense.prefill-f32-large"
        );
        assert_eq!(
            matmul_site(DType::BF16, 1, 512, crate::VOCAB, crate::HIDDEN, true).unwrap(),
            "dense.prefill-bf16-large-wide"
        );
    }
}
