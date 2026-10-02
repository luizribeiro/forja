use forja_sdk::{
    EngineLoadConfig, EngineSelections, Param, Result, VariantSelection, target::metal::Variant,
};

const SITES: &[&str] = &[
    "quant.decode-q4",
    "quant.decode-q8",
    "quant.prefill-small",
    "quant.prefill-tiled",
    "gather-quant-matmul.small",
    "gather-quant-matmul.grouped",
    "gather-quant-silu-mul.small",
    "gather-quant-silu-mul.grouped",
    "moe.combine-small",
    "moe.combine-large",
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

    pub(crate) fn quant(&self, rows: u32, bits: u8) -> Result<Variant> {
        self.0.at(quant_site(rows, bits), 0)
    }

    pub(crate) fn gather_matmul(&self, routed_rows: u32) -> Result<Variant> {
        self.gathered("gather-quant-matmul", routed_rows)
    }

    pub(crate) fn gather_silu_mul(&self, routed_rows: u32) -> Result<Variant> {
        self.gathered("gather-quant-silu-mul", routed_rows)
    }

    pub(crate) fn combine(&self, sequence: u32) -> Result<Variant> {
        let site = if u64::from(sequence) * u64::from(crate::HIDDEN) >= 1 << 20 {
            "moe.combine-large"
        } else {
            "moe.combine-small"
        };
        self.0.at(site, 0)
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

    fn gathered(&self, operation: &str, routed_rows: u32) -> Result<Variant> {
        let size = if routed_rows >= 64 {
            "grouped"
        } else {
            "small"
        };
        self.0.at(&format!("{operation}.{size}"), 0)
    }
}

fn quant_site(rows: u32, bits: u8) -> &'static str {
    if rows == 1 && bits == crate::Q8_BITS {
        "quant.decode-q8"
    } else if rows == 1 {
        "quant.decode-q4"
    } else if rows >= 16 {
        "quant.prefill-tiled"
    } else {
        "quant.prefill-small"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_quantized_dispatch_sites_at_boundaries() {
        assert_eq!(quant_site(1, crate::Q4_BITS), "quant.decode-q4");
        assert_eq!(quant_site(1, crate::Q8_BITS), "quant.decode-q8");
        assert_eq!(quant_site(15, crate::Q4_BITS), "quant.prefill-small");
        assert_eq!(quant_site(16, crate::Q4_BITS), "quant.prefill-tiled");
    }
}
