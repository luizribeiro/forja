use forja_sdk::{
    DType, EngineLoadConfig, EngineSelections, Param, Result, VariantSelection,
    target::metal::Variant,
};

const COMMON_SITES: &[&str] = &[
    "dense.decode",
    "attention.decode",
    "attention.prefill-large",
    "attention.prefill-small",
];
const F32_SITES: &[&str] = &["dense.prefill.large", "dense.prefill.small"];
const BF16_SITES: &[&str] = &["dense.prefill.large-wide", "dense.prefill.other"];

pub(crate) struct Variants(EngineSelections);

impl Variants {
    pub(crate) fn new(config: &EngineLoadConfig, dtype: DType) -> Result<Self> {
        let sites = if dtype == DType::BF16 {
            [COMMON_SITES, BF16_SITES].concat()
        } else {
            [COMMON_SITES, F32_SITES].concat()
        };
        Ok(Self(EngineSelections::new(
            config,
            crate::tuning::REGISTRY,
            &sites,
            crate::MAX_CONTEXT,
        )?))
    }

    pub(crate) fn matmul(
        &self,
        dtype: DType,
        rows: u32,
        inner: u32,
        columns: u32,
    ) -> Result<Variant> {
        self.0.at(matmul_site(dtype, rows, inner, columns), 0)
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

fn matmul_site(dtype: DType, rows: u32, inner: u32, columns: u32) -> &'static str {
    if rows == 1 {
        return "dense.decode";
    }
    let large = u64::from(rows)
        .checked_mul(u64::from(columns))
        .is_some_and(|elements| elements >= 1 << 20);
    if dtype == DType::BF16 {
        if large && u64::from(rows.max(columns)) * 2 > u64::from(inner) {
            "dense.prefill.large-wide"
        } else {
            "dense.prefill.other"
        }
    } else if large {
        "dense.prefill.large"
    } else {
        "dense.prefill.small"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_dense_dispatch_sites() {
        assert_eq!(
            matmul_site(DType::BF16, 1, crate::HIDDEN, crate::INTERMEDIATE),
            "dense.decode"
        );
        assert_eq!(
            matmul_site(DType::BF16, 512, crate::HIDDEN, crate::INTERMEDIATE),
            "dense.prefill.large-wide"
        );
        assert_eq!(
            matmul_site(
                DType::BF16,
                512,
                crate::HIDDEN,
                crate::KEY_VALUE_HEADS * crate::HEAD_DIM,
            ),
            "dense.prefill.other"
        );
        assert_eq!(
            matmul_site(DType::F32, 16, crate::HIDDEN, crate::INTERMEDIATE),
            "dense.prefill.small"
        );
    }
}
