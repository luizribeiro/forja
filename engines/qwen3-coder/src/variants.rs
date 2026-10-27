use forja_sdk::{
    EngineLoadConfig, EngineSelections, Param, Result, VariantSelection, target::metal::Variant,
};

use crate::reduction_classes::serving_reduction_class;

const SITES: &[&str] = &[
    "quant.vocab-decode",
    "quant.decode-q4",
    "quant.decode-q8",
    "quant.decode-b2-q4",
    "quant.decode-b2-q8",
    "quant.decode-b4-q4",
    "quant.decode-b4-q8",
    "quant.decode-b8-q4",
    "quant.decode-b8-q8",
    "quant.prefill-small",
    "quant.prefill-tiled",
    "gather-quant-matmul.small",
    "gather-quant-matmul.grouped-small-m",
    "gather-quant-matmul.grouped",
    "gather-quant-matmul.batch-2",
    "gather-quant-matmul.batch-4",
    "gather-quant-matmul.batch-8",
    "moe.gate-up.small",
    "moe.gate-up.grouped-small-m",
    "moe.gate-up.grouped",
    "moe.gate-up.batch-2",
    "moe.gate-up.batch-4",
    "moe.gate-up.batch-8",
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
    pub(crate) fn new(config: &EngineLoadConfig, max_context: u32) -> Result<Self> {
        validate_batch_reductions(config)?;
        validate_prefill_reductions(config)?;
        Ok(Self(EngineSelections::new(
            config,
            &[],
            SITES,
            max_context,
        )?))
    }

    pub(crate) fn quant(&self, rows: u32, bits: u8, columns: u32) -> Result<Variant> {
        self.0.at(quant_call_site(rows, bits, columns), 0)
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
        self.0.at(combine_site(sequence), 0)
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
        self.0.at(&gathered_site(operation, routed_rows), 0)
    }
}

fn validate_batch_reductions(config: &EngineLoadConfig) -> Result<()> {
    for sites in [
        &["quant.vocab-decode"][..],
        &[
            "quant.decode-q4",
            "quant.decode-b2-q4",
            "quant.decode-b4-q4",
            "quant.decode-b8-q4",
        ][..],
        &[
            "quant.decode-q8",
            "quant.decode-b2-q8",
            "quant.decode-b4-q8",
            "quant.decode-b8-q8",
        ],
        &[
            "gather-quant-matmul.small",
            "gather-quant-matmul.batch-2",
            "gather-quant-matmul.batch-4",
            "gather-quant-matmul.batch-8",
        ],
        &[
            "moe.gate-up.small",
            "moe.gate-up.batch-2",
            "moe.gate-up.batch-4",
            "moe.gate-up.batch-8",
        ],
    ] {
        validate_reduction_classes(config, sites, "decode")?;
    }
    Ok(())
}

fn validate_prefill_reductions(config: &EngineLoadConfig) -> Result<()> {
    for sites in [
        &["quant.prefill-tiled"][..],
        &["gather-quant-matmul.grouped"],
        &["moe.gate-up.grouped"],
        &["moe.combine-large"],
        &["attention.prefill-small", "attention.prefill-large"],
        &[
            "paged-attention.prefill-small",
            "paged-attention.prefill-large",
        ],
    ] {
        validate_reduction_classes(config, sites, "prefill")?;
    }
    Ok(())
}

fn validate_reduction_classes(
    config: &EngineLoadConfig,
    sites: &[&str],
    phase: &str,
) -> Result<()> {
    let mut class = None;
    for &site in sites {
        let names = config
            .fixed_variant_picks
            .iter()
            .filter(|pick| pick.site == site)
            .map(|pick| pick.name.as_str())
            .chain(
                config
                    .variant_rule_picks
                    .iter()
                    .filter(|rule| rule.site == site)
                    .flat_map(|rule| rule.arms.iter().map(|arm| arm.name.as_str())),
            );
        for name in names {
            let next = serving_reduction_class(name).ok_or_else(|| {
                forja_sdk::Error::loading(format!(
                    "serving site {site:?} selected a non-invariant variant"
                ))
            })?;
            if class.replace(next).is_some_and(|class| class != next) {
                return Err(forja_sdk::Error::loading(format!(
                    "serving {phase} chunks require one reduction class"
                )));
            }
        }
    }
    Ok(())
}

fn combine_site(sequence: u32) -> &'static str {
    if sequence >= crate::PREFILL_CHUNK_MULTIPLE {
        "moe.combine-large"
    } else {
        "moe.combine-small"
    }
}

fn gathered_site(operation: &str, routed_rows: u32) -> String {
    match routed_rows {
        16 => format!("{operation}.batch-2"),
        32 => format!("{operation}.batch-4"),
        64 => format!("{operation}.batch-8"),
        _ => format!("{operation}.{}", gathered_size(routed_rows)),
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

fn quant_call_site(rows: u32, bits: u8, columns: u32) -> &'static str {
    if rows <= 8 && columns == crate::VOCAB {
        "quant.vocab-decode"
    } else {
        quant_site(rows, bits)
    }
}

fn quant_site(rows: u32, bits: u8) -> &'static str {
    match (rows, bits) {
        (1, crate::Q8_BITS) => "quant.decode-q8",
        (1, _) => "quant.decode-q4",
        (2, crate::Q8_BITS) => "quant.decode-b2-q8",
        (2, _) => "quant.decode-b2-q4",
        (4, crate::Q8_BITS) => "quant.decode-b4-q8",
        (4, _) => "quant.decode-b4-q4",
        (8, crate::Q8_BITS) => "quant.decode-b8-q8",
        (8, _) => "quant.decode-b8-q4",
        (16.., _) => "quant.prefill-tiled",
        _ => "quant.prefill-small",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_decode_keeps_one_pick_across_batch_buckets() {
        for rows in [1, 2, 4, 7, 8] {
            assert_eq!(
                quant_call_site(rows, crate::Q4_BITS, crate::VOCAB),
                "quant.vocab-decode"
            );
            assert_eq!(
                quant_call_site(rows, crate::Q4_BITS, 5120),
                quant_site(rows, crate::Q4_BITS)
            );
        }
        assert_eq!(
            quant_call_site(16, crate::Q4_BITS, crate::VOCAB),
            "quant.prefill-tiled"
        );
    }

    #[test]
    fn classifies_quantized_dispatch_sites_at_boundaries() {
        assert_eq!(quant_site(1, crate::Q4_BITS), "quant.decode-q4");
        assert_eq!(quant_site(1, crate::Q8_BITS), "quant.decode-q8");
        assert_eq!(quant_site(2, crate::Q4_BITS), "quant.decode-b2-q4");
        assert_eq!(quant_site(4, crate::Q8_BITS), "quant.decode-b4-q8");
        assert_eq!(quant_site(8, crate::Q4_BITS), "quant.decode-b8-q4");
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
    fn assigns_routed_decode_rows_to_batch_sites() {
        assert_eq!(gathered_site("moe.gate-up", 16), "moe.gate-up.batch-2");
        assert_eq!(gathered_site("moe.gate-up", 32), "moe.gate-up.batch-4");
        assert_eq!(gathered_site("moe.gate-up", 64), "moe.gate-up.batch-8");
    }

    #[test]
    fn distinguishes_cross_bucket_reduction_classes() {
        assert_eq!(
            serving_reduction_class("quant-matmul.gemv"),
            serving_reduction_class("quant-matmul.row-gemv")
        );
        assert_eq!(
            serving_reduction_class("quant-matmul.gemv-n2"),
            serving_reduction_class("quant-matmul.row-gemv")
        );
        assert_eq!(
            serving_reduction_class("quant-matmul.q8-fast-gemv-n2"),
            serving_reduction_class("quant-matmul.q8-fast-gemv")
        );
        assert_ne!(
            serving_reduction_class("gather-quant-matmul.route-gemv"),
            serving_reduction_class("gather-quant-matmul.grouped-small-m")
        );
        assert_eq!(serving_reduction_class("matmul.portable"), None);
    }

    #[test]
    fn prefill_accepts_unblocked_expert_reductions() {
        let config = EngineLoadConfig {
            fixed_variant_picks: ["gather-quant-matmul.grouped", "moe.gate-up.grouped"]
                .into_iter()
                .map(|site| forja_sdk::FixedVariantPick {
                    site: site.into(),
                    name: "gather-quant-matmul.grouped-32x32-2x2".into(),
                })
                .collect(),
            ..EngineLoadConfig::default()
        };
        assert!(validate_prefill_reductions(&config).is_ok());
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
    #[test]
    fn prefill_rules_require_one_known_reduction_class() {
        use forja_sdk::{VariantPickArm, VariantRulePick};
        for (small, large, shared, different) in [
            (
                "paged-attention.prefill-small",
                "paged-attention.prefill-large",
                "paged-sdpa.steel-half-64x16",
                "paged-sdpa.steel-half-32x16",
            ),
            (
                "attention.prefill-small",
                "attention.prefill-large",
                "sdpa.steel-half",
                "sdpa.steel",
            ),
        ] {
            let mut config = EngineLoadConfig {
                fixed_variant_picks: vec![forja_sdk::FixedVariantPick {
                    site: small.into(),
                    name: different.into(),
                }],
                variant_rule_picks: vec![VariantRulePick {
                    site: large.into(),
                    parameter: "position".into(),
                    arms: [shared, shared]
                        .into_iter()
                        .enumerate()
                        .map(|(index, name)| VariantPickArm {
                            lo: u32::try_from(index).unwrap() * 2048,
                            hi: u32::try_from(index).unwrap() * 2048 + 2047,
                            name: name.into(),
                        })
                        .collect(),
                }],
                ..EngineLoadConfig::default()
            };
            assert!(validate_prefill_reductions(&config).is_err());
            config.fixed_variant_picks[0].name = shared.into();
            assert!(validate_prefill_reductions(&config).is_ok());
            config.variant_rule_picks[0].arms[1].name = different.into();
            assert!(validate_prefill_reductions(&config).is_err());
            config.variant_rule_picks[0].arms[1].name = "matmul.portable".into();
            assert!(validate_prefill_reductions(&config).is_err());
        }
    }

    #[test]
    fn mergeable_prefill_chunks_use_validated_sites() {
        for rows in [64, 128, 192, 256, 448, 512, 704, 1024, 1536, 2048] {
            for bits in [crate::Q4_BITS, crate::Q8_BITS] {
                assert_eq!(quant_site(rows, bits), "quant.prefill-tiled");
            }
            assert_eq!(gathered_size(rows * crate::TOP_K), "grouped");
            assert!(["prefill-small", "prefill-large"].contains(&attention_phase(rows, true)));
            assert_eq!(combine_site(rows), "moe.combine-large");
        }
    }
    #[test]
    fn combine_keeps_decode_buckets_outside_prefill_alignment() {
        for rows in [1, 2, 4, 8, 63] {
            assert_eq!(combine_site(rows), "moe.combine-small");
        }
        assert_eq!(combine_site(64), "moe.combine-large");
    }
}
