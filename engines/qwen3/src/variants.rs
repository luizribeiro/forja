use forja_sdk::{
    DType, EngineLoadConfig, EngineSelections, Param, Result, VariantSelection,
    target::metal::Variant,
};

const COMMON_SITES: &[&str] = &[
    "dense.decode",
    "sample",
    "attention.decode",
    "attention.prefill-large",
    "attention.prefill-small",
    "paged-attention.decode",
    "paged-attention.prefill-large",
    "paged-attention.prefill-small",
    "paged-attention.prefill-tiny",
];
const F32_SITES: &[&str] = &["dense.prefill.large", "dense.prefill.small"];
const BF16_SITES: &[&str] = &["dense.prefill.large-wide", "dense.prefill.other"];

pub(crate) struct Variants(EngineSelections);

impl Variants {
    pub(crate) fn new(config: &EngineLoadConfig, dtype: DType, max_context: u32) -> Result<Self> {
        let sites = if dtype == DType::BF16 {
            [COMMON_SITES, BF16_SITES].concat()
        } else {
            [COMMON_SITES, F32_SITES].concat()
        };
        Ok(Self(EngineSelections::new(
            config,
            &[],
            &sites,
            max_context,
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

    pub(crate) fn sample(&self) -> Result<Variant> {
        self.0.at("sample", 0)
    }

    pub(crate) fn attention(
        &self,
        sequence: u32,
        start: u32,
        parameter: Option<&Param>,
        paged: bool,
    ) -> Result<VariantSelection> {
        let phase = attention_phase(sequence, paged);
        let site = if paged {
            format!("paged-attention.{phase}")
        } else {
            format!("attention.{phase}")
        };
        if let Some(parameter) = parameter {
            self.0.for_param(&site, parameter)
        } else {
            Ok(VariantSelection::Fixed(self.0.at(&site, start)?))
        }
    }
}

fn attention_phase(sequence: u32, paged: bool) -> &'static str {
    if sequence == 1 {
        "decode"
    } else if sequence >= 512 {
        "prefill-large"
    } else if paged && sequence <= 33 {
        "prefill-tiny"
    } else {
        "prefill-small"
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

    #[test]
    fn classifies_paged_attention_sites_at_boundaries() {
        assert_eq!(attention_phase(1, true), "decode");
        assert_eq!(attention_phase(2, true), "prefill-tiny");
        assert_eq!(attention_phase(33, true), "prefill-tiny");
        assert_eq!(attention_phase(34, true), "prefill-small");
        assert_eq!(attention_phase(511, true), "prefill-small");
        assert_eq!(attention_phase(512, true), "prefill-large");
        assert_eq!(attention_phase(33, false), "prefill-small");
    }
}
