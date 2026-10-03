use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Profile, VariantRule, Variants};

/// Engine-owned development overrides.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Engine {
    /// Whether the guest captures and replays reusable graphs.
    pub replay: bool,
    /// Optional tuning-set edits.
    pub tunings: TuningOverrides,
    /// Fixed variant overrides keyed by dispatch site.
    pub variant_picks: BTreeMap<String, String>,
    /// Parameterized variant overrides keyed by dispatch site.
    pub variant_rules: BTreeMap<String, VariantRule>,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            replay: true,
            tunings: TuningOverrides::default(),
            variant_picks: BTreeMap::new(),
            variant_rules: BTreeMap::new(),
        }
    }
}

/// A tuning set expressed relative to profile defaults or the empty set.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct TuningOverrides {
    /// Starting tuning set.
    pub base: TuningBase,
    /// Tunings added to the starting set.
    pub add: Vec<String>,
    /// Tunings removed from the starting set.
    pub remove: Vec<String>,
}

impl Default for TuningOverrides {
    fn default() -> Self {
        Self {
            base: TuningBase::Profile,
            add: Vec::new(),
            remove: Vec::new(),
        }
    }
}

/// Source set for tuning overrides.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum TuningBase {
    /// Begin with the profile defaults.
    #[default]
    Profile,
    /// Begin with no tunings.
    None,
}

/// Effective engine choices after applying validated overrides.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineSelection {
    /// Effective tuning names.
    pub tunings: Vec<String>,
    /// Effective variant selections.
    pub variants: Variants,
}

impl Engine {
    /// Applies these overrides to an engine profile.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate or contradictory tuning edits, unknown removals or sites,
    /// and malformed variant rules.
    pub fn resolve(&self, profile: &Profile) -> Result<EngineSelection, String> {
        let mut tunings = match self.tunings.base {
            TuningBase::Profile => profile.default_tunings().to_vec(),
            TuningBase::None => Vec::new(),
        };
        validate_unique("engine.tunings.add", &self.tunings.add)?;
        validate_unique("engine.tunings.remove", &self.tunings.remove)?;
        if let Some(name) = self
            .tunings
            .add
            .iter()
            .find(|name| self.tunings.remove.contains(name))
        {
            return Err(format!("engine tuning {name:?} is both added and removed"));
        }
        for name in &self.tunings.add {
            validate_name("engine tuning", name)?;
            if tunings.contains(name) {
                return Err(format!("engine tuning {name:?} is already selected"));
            }
            tunings.push(name.clone());
        }
        for name in &self.tunings.remove {
            let Some(index) = tunings.iter().position(|selected| selected == name) else {
                return Err(format!("engine tuning {name:?} is not selected"));
            };
            tunings.remove(index);
        }

        let mut variants = profile.variants().clone();
        for tuning in &tunings {
            if let Some(overrides) = profile.tuning_variants().get(tuning) {
                overlay(&mut variants, overrides);
            }
        }
        let known_sites = profile_sites(profile);
        for (site, name) in &self.variant_picks {
            require_site(&known_sites, site)?;
            validate_name("engine variant", name)?;
            variants.rules.remove(site);
            variants.fixed.insert(site.clone(), name.clone());
        }
        for (site, rule) in &self.variant_rules {
            require_site(&known_sites, site)?;
            validate_rule(profile, site, rule)?;
            variants.fixed.remove(site);
            variants.rules.insert(site.clone(), rule.clone());
        }
        if let Some(site) = self
            .variant_picks
            .keys()
            .find(|site| self.variant_rules.contains_key(*site))
        {
            return Err(format!(
                "engine variant site {site:?} has both a pick and a rule"
            ));
        }
        Ok(EngineSelection { tunings, variants })
    }
}

fn overlay(base: &mut Variants, overrides: &Variants) {
    for (site, name) in &overrides.fixed {
        base.rules.remove(site);
        base.fixed.insert(site.clone(), name.clone());
    }
    for (site, rule) in &overrides.rules {
        base.fixed.remove(site);
        base.rules.insert(site.clone(), rule.clone());
    }
}

fn profile_sites(profile: &Profile) -> BTreeSet<&str> {
    profile
        .variants()
        .fixed
        .keys()
        .chain(profile.variants().rules.keys())
        .chain(
            profile
                .tuning_variants()
                .values()
                .flat_map(|variants| variants.fixed.keys().chain(variants.rules.keys())),
        )
        .map(String::as_str)
        .collect()
}

fn require_site(sites: &BTreeSet<&str>, site: &str) -> Result<(), String> {
    if sites.contains(site) {
        Ok(())
    } else {
        Err(format!("unknown engine variant site {site:?}"))
    }
}

fn validate_unique(label: &str, names: &[String]) -> Result<(), String> {
    let unique = names.iter().collect::<BTreeSet<_>>();
    if unique.len() == names.len() {
        Ok(())
    } else {
        Err(format!("{label} must be unique"))
    }
}

fn validate_name(label: &str, name: &str) -> Result<(), String> {
    if name.is_empty() || name.len() > 64 {
        Err(format!("{label} must contain 1..=64 bytes"))
    } else {
        Ok(())
    }
}

fn validate_rule(profile: &Profile, site: &str, rule: &VariantRule) -> Result<(), String> {
    validate_name("engine variant parameter", &rule.parameter)?;
    let mut next = 0;
    for arm in &rule.arms {
        validate_name("engine variant", &arm.name)?;
        if arm.lo != next || arm.hi < arm.lo {
            return Err(format!(
                "engine variant rule {site:?} has incomplete or overlapping arms"
            ));
        }
        next = arm
            .hi
            .checked_add(1)
            .ok_or_else(|| format!("engine variant rule {site:?} range overflowed"))?;
    }
    if next != profile.workload().max_context.get() {
        return Err(format!(
            "engine variant rule {site:?} does not cover the profile context"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use super::*;

    fn profile() -> Profile {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../engines/qwen3/profiles/qwen3-0.6b.metal-apple-m3-ultra.bf16.toml");
        toml::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn parses_typed_engine_overrides() {
        let config: crate::DevConfig = toml::from_str(
            r#"
[engine.tunings]
base = "none"
add = ["residual-norm"]

[engine.variant-picks]
"dense.decode" = "matmul.portable"
"#,
        )
        .unwrap();
        assert!(config.engine.replay);
        assert_eq!(config.engine.tunings.base, TuningBase::None);
        assert_eq!(config.engine.tunings.add, ["residual-norm"]);
        assert_eq!(
            config.engine.variant_picks["dense.decode"],
            "matmul.portable"
        );
    }

    #[test]
    fn parses_lazy_execution_override() {
        let config: crate::DevConfig = toml::from_str("[engine]\nreplay = false\n").unwrap();
        assert!(!config.engine.replay);
    }

    #[test]
    fn resolves_tunings_and_variant_picks_against_profile() {
        let mut overrides = Engine::default();
        overrides.tunings.remove.push("silu-mul".to_owned());
        overrides.variant_picks.insert(
            "attention.prefill-small".to_owned(),
            "sdpa.portable".to_owned(),
        );
        let selection = overrides.resolve(&profile()).unwrap();
        assert!(!selection.tunings.contains(&"silu-mul".to_owned()));
        assert_eq!(
            selection.variants.fixed["attention.prefill-small"],
            "sdpa.portable"
        );
    }

    #[test]
    fn rejects_contradictory_tunings_and_unknown_sites() {
        let mut overrides = Engine::default();
        overrides.tunings.base = TuningBase::None;
        overrides.tunings.add.push("final-norm".to_owned());
        overrides.tunings.remove.push("final-norm".to_owned());
        assert!(overrides.resolve(&profile()).is_err());

        let mut overrides = Engine::default();
        overrides
            .variant_picks
            .insert("unknown".to_owned(), "matmul.portable".to_owned());
        assert!(overrides.resolve(&profile()).is_err());
    }
}
