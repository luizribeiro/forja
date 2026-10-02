//! Validation and lookup for engine-owned tuning and variant registries.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    EngineLoadConfig, Error, Result, VariantRulePick,
    graph::Param,
    target::metal::{Variant, VariantChoice, VariantRule},
};

/// Static metadata for one engine tuning module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Tuning {
    /// Stable tuning name.
    pub name: &'static str,
    /// Engine-owned replacement slot.
    pub slot: &'static str,
    /// Target device family.
    pub device_family: &'static str,
    /// Shape class for which the tuning was written.
    pub shape_class: &'static str,
    /// User-facing summary.
    pub description: &'static str,
    /// Reason the tuning exists.
    pub rationale: &'static str,
}

/// One owned algorithm selection resolved from an engine dispatch site.
pub enum VariantSelection {
    /// A fixed backend algorithm.
    Fixed(Variant),
    /// A parameter-dependent backend rule.
    Rule(VariantRule),
}

impl VariantSelection {
    /// Borrows the selection for an operation call.
    #[must_use]
    pub fn choice(&self) -> VariantChoice<'_> {
        match self {
            Self::Fixed(variant) => variant.into(),
            Self::Rule(rule) => rule.into(),
        }
    }
}

/// Validated selections for one engine instance.
pub struct EngineSelections {
    tunings: EngineTunings,
    fixed: BTreeMap<String, Variant>,
    rules: BTreeMap<String, VariantRulePick>,
}

/// Validated active tuning names for one engine instance.
pub struct EngineTunings(BTreeSet<String>);

impl EngineTunings {
    /// Validates effective tuning names against an engine registry.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown or duplicate name or two tunings selecting one slot.
    pub fn new(config: &EngineLoadConfig, registry: &[Tuning]) -> Result<Self> {
        let known = registry
            .iter()
            .map(|tuning| (tuning.name, tuning))
            .collect::<BTreeMap<_, _>>();
        let mut tunings = BTreeSet::new();
        let mut slots = BTreeSet::new();
        for name in &config.tunings {
            if !tunings.insert(name.clone()) {
                return Err(Error::loading(format!("duplicate tuning {name:?}")));
            }
            let tuning = known
                .get(name.as_str())
                .ok_or_else(|| Error::loading(format!("unknown tuning {name:?}")))?;
            if !slots.insert(tuning.slot) {
                return Err(Error::loading(format!(
                    "multiple tunings select slot {:?}",
                    tuning.slot
                )));
            }
        }
        Ok(Self(tunings))
    }

    /// Returns whether a registered tuning is active.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.0.contains(name)
    }
}

impl EngineSelections {
    /// Validates effective load selections against an engine registry.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown or duplicate tunings and sites, two tunings for one slot,
    /// missing picks, or malformed fixed/rule selections.
    pub fn new(
        config: &EngineLoadConfig,
        registry: &[Tuning],
        sites: &[&str],
        max_context: u32,
    ) -> Result<Self> {
        let tunings = EngineTunings::new(config, registry)?;
        let known_sites = sites.iter().copied().collect::<BTreeSet<_>>();
        let mut fixed = BTreeMap::new();
        for pick in &config.fixed_variant_picks {
            require_site(&known_sites, &pick.site)?;
            let variant = Variant::new(pick.name.clone())?;
            if fixed.insert(pick.site.clone(), variant).is_some() {
                return Err(Error::loading(format!(
                    "duplicate variant pick for site {:?}",
                    pick.site
                )));
            }
        }
        let mut rules = BTreeMap::new();
        for pick in &config.variant_rule_picks {
            require_site(&known_sites, &pick.site)?;
            if fixed.contains_key(&pick.site) || rules.contains_key(&pick.site) {
                return Err(Error::loading(format!(
                    "duplicate variant pick for site {:?}",
                    pick.site
                )));
            }
            validate_rule(pick, max_context)?;
            rules.insert(pick.site.clone(), pick.clone());
        }
        if let Some(site) = sites
            .iter()
            .find(|site| !fixed.contains_key(**site) && !rules.contains_key(**site))
        {
            return Err(Error::loading(format!(
                "missing variant pick for site {site:?}"
            )));
        }
        Ok(Self {
            tunings,
            fixed,
            rules,
        })
    }

    /// Returns whether a registered tuning is active.
    #[must_use]
    pub fn tuning(&self, name: &str) -> bool {
        self.tunings.contains(name)
    }

    /// Returns one fixed selection or evaluates a rule at a concrete position.
    ///
    /// # Errors
    ///
    /// Returns an error when the site is absent or its rule has no matching arm.
    pub fn at(&self, site: &str, position: u32) -> Result<Variant> {
        if let Some(variant) = self.fixed.get(site) {
            return Ok(variant.clone());
        }
        let rule = self
            .rules
            .get(site)
            .ok_or_else(|| Error::loading(format!("unknown variant site {site:?}")))?;
        rule.arms
            .iter()
            .find(|arm| arm.lo <= position && position <= arm.hi)
            .map(|arm| Variant::new(arm.name.clone()))
            .transpose()?
            .ok_or_else(|| Error::loading(format!("variant rule {site:?} has no matching arm")))
    }

    /// Returns a fixed selection or a rule clipped to a capture parameter's range.
    ///
    /// # Errors
    ///
    /// Returns an error when the site is absent or the clipped rule is incomplete.
    pub fn for_param(&self, site: &str, parameter: &Param) -> Result<VariantSelection> {
        if let Some(variant) = self.fixed.get(site) {
            return Ok(VariantSelection::Fixed(variant.clone()));
        }
        let rule = self
            .rules
            .get(site)
            .ok_or_else(|| Error::loading(format!("unknown variant site {site:?}")))?;
        let start = *parameter.range().start();
        let end = *parameter.range().end();
        let arms = rule
            .arms
            .iter()
            .filter_map(|arm| {
                let lo = arm.lo.max(start);
                let hi = arm.hi.min(end);
                (lo <= hi).then(|| Variant::new(arm.name.clone()).map(|name| (lo..=hi, name)))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(VariantSelection::Rule(VariantRule::new(parameter, arms)?))
    }
}

fn require_site(known: &BTreeSet<&str>, site: &str) -> Result<()> {
    if known.contains(site) {
        Ok(())
    } else {
        Err(Error::loading(format!("unknown variant site {site:?}")))
    }
}

fn validate_rule(rule: &VariantRulePick, max_context: u32) -> Result<()> {
    if rule.parameter != "position" || max_context == 0 {
        return Err(Error::loading(format!(
            "invalid variant rule for site {:?}",
            rule.site
        )));
    }
    let mut next = 0_u64;
    for arm in &rule.arms {
        Variant::new(arm.name.clone())?;
        if u64::from(arm.lo) != next || arm.lo > arm.hi {
            return Err(Error::loading(format!(
                "invalid variant rule for site {:?}",
                rule.site
            )));
        }
        next = u64::from(arm.hi) + 1;
    }
    if next != u64::from(max_context) {
        return Err(Error::loading(format!(
            "invalid variant rule for site {:?}",
            rule.site
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FixedVariantPick, VariantPickArm};

    const TUNINGS: &[Tuning] = &[
        Tuning {
            name: "fused-a",
            slot: "norm",
            device_family: "apple-m3-ultra",
            shape_class: "hidden-1024",
            description: "a",
            rationale: "a",
        },
        Tuning {
            name: "fused-b",
            slot: "norm",
            device_family: "apple-m3-ultra",
            shape_class: "hidden-1024",
            description: "b",
            rationale: "b",
        },
    ];

    fn config() -> EngineLoadConfig {
        EngineLoadConfig {
            tunings: vec!["fused-a".to_owned()],
            fixed_variant_picks: vec![FixedVariantPick {
                site: "dense.decode".to_owned(),
                name: "matmul.gemv-transposed".to_owned(),
            }],
            variant_rule_picks: vec![VariantRulePick {
                site: "attention.decode".to_owned(),
                parameter: "position".to_owned(),
                arms: vec![VariantPickArm {
                    lo: 0,
                    hi: 31,
                    name: "sdpa.decomposed".to_owned(),
                }],
            }],
            ..EngineLoadConfig::default()
        }
    }

    #[test]
    fn validates_registered_tunings_and_picks() {
        let selections = EngineSelections::new(
            &config(),
            TUNINGS,
            &["dense.decode", "attention.decode"],
            32,
        )
        .unwrap();
        assert!(selections.tuning("fused-a"));
        assert_eq!(
            selections.at("attention.decode", 31).unwrap().name(),
            "sdpa.decomposed"
        );
    }

    #[test]
    fn rejects_unknown_duplicate_and_conflicting_tunings() {
        let mut unknown = config();
        unknown.tunings = vec!["unknown".to_owned()];
        assert!(EngineSelections::new(&unknown, TUNINGS, &["dense.decode"], 32).is_err());

        let mut duplicate = config();
        duplicate.tunings.push("fused-a".to_owned());
        assert!(EngineSelections::new(&duplicate, TUNINGS, &["dense.decode"], 32).is_err());

        let mut conflicting = config();
        conflicting.tunings.push("fused-b".to_owned());
        assert!(EngineSelections::new(&conflicting, TUNINGS, &["dense.decode"], 32).is_err());
    }

    #[test]
    fn rejects_missing_unknown_duplicate_and_invalid_picks() {
        let mut missing = config();
        missing.fixed_variant_picks.clear();
        assert!(
            EngineSelections::new(&missing, TUNINGS, &["dense.decode", "attention.decode"], 32)
                .is_err()
        );

        let mut unknown = config();
        unknown.fixed_variant_picks[0].site = "unknown".to_owned();
        assert!(EngineSelections::new(&unknown, TUNINGS, &["dense.decode"], 32).is_err());

        let mut duplicate = config();
        duplicate
            .fixed_variant_picks
            .push(duplicate.fixed_variant_picks[0].clone());
        assert!(EngineSelections::new(&duplicate, TUNINGS, &["dense.decode"], 32).is_err());

        let mut invalid = config();
        invalid.fixed_variant_picks[0].name = "not-a-variant".to_owned();
        assert!(EngineSelections::new(&invalid, TUNINGS, &["dense.decode"], 32).is_err());
    }
}
