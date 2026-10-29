//! The engine's profile-controlled optimization and dispatch registry.

#![forbid(unsafe_code)]

use forja_sdk::{
    EngineLoadConfig, EngineSelections, Param, Result, Tuning, VariantSelection,
    target::metal::Variant,
};

/// Registers the optimized implementation without a build feature.
pub const TUNINGS: &[Tuning] = &[
    Tuning {
        name: "optimized",
        slot: "execution",
        device_family: "apple-m3-ultra",
        shape_class: "llama",
        description: "Use engine-owned optimized execution",
        rationale: "Keep the readable implementation available in the same component",
    },
    Tuning {
        name: "fused-epilogues",
        slot: "fusion",
        device_family: "apple-m3-ultra",
        shape_class: "llama",
        description: "Fuse the gated up projection and residual normalization",
        rationale: "Avoid intermediate activation writes and separate dispatches",
    },
];

/// Validated profile choices for the optimized implementation.
pub struct Variants(EngineSelections);

impl Variants {
    /// Resolves every dispatch site before model execution.
    ///
    /// # Errors
    /// Returns an error for unknown tunings, missing picks or incomplete rules.
    pub fn new(config: &EngineLoadConfig, context: u32) -> Result<Self> {
        Ok(Self(EngineSelections::new(
            config,
            TUNINGS,
            &[
                "dense.decode",
                "dense.prefill",
                "attention.decode",
                "attention.prefill",
                "paged-attention.decode",
                "paged-attention.prefill",
            ],
            context,
        )?))
    }

    /// Chooses a matrix algorithm for one row or multiple prompt rows.
    ///
    /// # Errors
    /// Returns an error when the selected dispatch site is unavailable.
    pub fn linear(&self, rows: u32) -> Result<Variant> {
        self.0.at(
            if rows == 1 {
                "dense.decode"
            } else {
                "dense.prefill"
            },
            0,
        )
    }

    /// Resolves a causal attention pick, retaining position rules in graph capture.
    ///
    /// # Errors
    /// Returns an error for missing picks or incomplete position rules.
    pub fn attention(
        &self,
        sequence: u32,
        start: u32,
        parameter: Option<&Param>,
        paged: bool,
    ) -> Result<VariantSelection> {
        let site = match (paged, sequence) {
            (true, 1) => "paged-attention.decode",
            (true, _) => "paged-attention.prefill",
            (false, 1) => "attention.decode",
            (false, _) => "attention.prefill",
        };
        parameter.map_or_else(
            || Ok(VariantSelection::Fixed(self.0.at(site, start)?)),
            |parameter| self.0.for_param(site, parameter),
        )
    }
}
