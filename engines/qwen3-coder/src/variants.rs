use forja_sdk::{
    EngineLoadConfig, EngineSelections, Param, Result, VariantSelection, target::metal::Variant,
};

const SITES: &[&str] = &[
    "quant.decode-q4",
    "quant.decode-q8",
    "quant.prefill-small",
    "quant.prefill-tiled",
    "gather-quant-matmul.small",
    "gather-quant-matmul.grouped-small-m",
    "gather-quant-matmul.grouped",
    "moe.gate-up.small",
    "moe.gate-up.grouped-small-m",
    "moe.gate-up.grouped",
    "moe.down-combine",
    "moe.combine-small",
    "moe.combine-large",
    "top-k",
    "sample",
    "attention.decode",
    "attention.prefill-large",
    "attention.prefill-small",
    "paged-attention.decode",
    "paged-attention.prefill-large",
    "paged-attention.prefill-small",
    "paged-attention.prefill-tiny",
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

    pub(crate) fn gate_up(&self, routed_rows: u32) -> Result<Variant> {
        self.gathered("moe.gate-up", routed_rows)
    }

    pub(crate) fn down_combine(&self) -> Result<Variant> {
        self.0.at("moe.down-combine", 0)
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

    fn gathered(&self, operation: &str, routed_rows: u32) -> Result<Variant> {
        self.0
            .at(&format!("{operation}.{}", gathered_size(routed_rows)), 0)
    }
}

fn gathered_size(routed_rows: u32) -> &'static str {
    if routed_rows >= 512 {
        "grouped"
    } else if routed_rows >= 64 {
        "grouped-small-m"
    } else {
        "small"
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

    #[test]
    fn classifies_grouped_dispatch_sites_at_boundaries() {
        assert_eq!(gathered_size(63), "small");
        assert_eq!(gathered_size(64), "grouped-small-m");
        assert_eq!(gathered_size(511), "grouped-small-m");
        assert_eq!(gathered_size(512), "grouped");
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
