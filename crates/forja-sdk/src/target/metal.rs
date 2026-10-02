//! Explicit Metal algorithm selections.

use std::ops::RangeInclusive;

use crate::{Error, Result, graph::Param};

const MAX_VARIANT_NAME_BYTES: usize = 64;
const MAX_VARIANT_RULE_ARMS: usize = 64;

/// One stable Metal algorithm name.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Variant {
    name: String,
}

impl Variant {
    /// Validates and retains one stable algorithm name without resolving it.
    ///
    /// # Errors
    ///
    /// Returns an error unless the name is two lowercase ASCII segments separated by a period,
    /// at most 64 bytes, with internal digits and hyphens accepted.
    pub fn new(name: impl Into<String>) -> Result<Self> {
        let name = name.into();
        let mut parts = name.split('.');
        let valid_part = |part: &str| {
            !part.is_empty()
                && !part.starts_with('-')
                && !part.ends_with('-')
                && part
                    .bytes()
                    .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        };
        let valid = name.len() <= MAX_VARIANT_NAME_BYTES
            && parts.next().is_some_and(valid_part)
            && parts.next().is_some_and(valid_part)
            && parts.next().is_none();
        if !valid {
            return Err(Error::new("invalid Metal variant name"));
        }
        Ok(Self { name })
    }

    /// Returns the retained stable name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// One inclusive parameter partition selecting explicit Metal algorithms.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariantRule {
    pub(crate) parameter_id: u64,
    pub(crate) arms: Vec<crate::sys::VariantArm>,
}

impl VariantRule {
    /// Validates ordered arms that exactly cover the parameter's declared range.
    ///
    /// This checks only rule and name syntax. Algorithm availability and constraints are resolved
    /// by the Metal host when the operation is recorded.
    ///
    /// # Errors
    ///
    /// Returns an error for no arms, an empty interval, a gap, overlap, ordering error, or an
    /// incomplete final arm.
    pub fn new(parameter: &Param, arms: Vec<(RangeInclusive<u32>, Variant)>) -> Result<Self> {
        if arms.is_empty() {
            return Err(Error::new("Metal variant rule requires at least one arm"));
        }
        if arms.len() > MAX_VARIANT_RULE_ARMS {
            return Err(Error::new("Metal variant rule has too many arms"));
        }
        let mut expected = *parameter.range().start();
        let arm_count = arms.len();
        let mut validated = Vec::with_capacity(arm_count);
        for (index, (range, variant)) in arms.into_iter().enumerate() {
            if range.is_empty() {
                return Err(Error::new("Metal variant rule arm cannot be empty"));
            }
            if *range.start() != expected {
                return Err(Error::new(
                    "Metal variant rule arms must be ordered and exactly cover the parameter",
                ));
            }
            let hi = *range.end();
            validated.push(crate::sys::VariantArm {
                lo: expected,
                hi,
                name: variant.name,
            });
            if index + 1 < arm_count {
                expected = hi.checked_add(1).ok_or_else(|| {
                    Error::new(
                        "Metal variant rule arms must be ordered and exactly cover the parameter",
                    )
                })?;
            }
        }
        if validated.last().map(|arm| arm.hi) != Some(*parameter.range().end()) {
            return Err(Error::new(
                "Metal variant rule arms must cover the complete parameter range",
            ));
        }
        Ok(Self {
            parameter_id: parameter.id(),
            arms: validated,
        })
    }
}

/// A borrowed explicit selection accepted by Metal-specific operation methods.
#[doc(hidden)]
#[derive(Clone, Copy)]
pub enum VariantChoice<'a> {
    /// One fixed algorithm.
    Variant(&'a Variant),
    /// A replay-parameter rule.
    Rule(&'a VariantRule),
}

impl<'a> From<&'a Variant> for VariantChoice<'a> {
    fn from(variant: &'a Variant) -> Self {
        Self::Variant(variant)
    }
}

impl<'a> From<&'a VariantRule> for VariantChoice<'a> {
    fn from(rule: &'a VariantRule) -> Self {
        Self::Rule(rule)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructors_validate_syntax_without_resolving_names() {
        assert!(Variant::new("future-op.unknown-algorithm").is_ok());
        for invalid in [
            "",
            "matmul",
            "Matmul.gemv",
            ".gemv",
            "matmul.",
            "matmul..gemv",
            "matmul.-gemv",
        ] {
            assert!(Variant::new(invalid).is_err(), "accepted {invalid:?}");
        }

        let parameter = Param::new(7..=33).unwrap();
        let low = Variant::new("future.low").unwrap();
        let high = Variant::new("future.high").unwrap();
        assert!(
            VariantRule::new(
                &parameter,
                vec![(7..=7, low.clone()), (8..=33, high.clone())]
            )
            .is_ok()
        );
        assert!(VariantRule::new(&parameter, vec![(7..=7, low), (9..=33, high)]).is_err());

        assert!(Variant::new(format!("future.{}", "a".repeat(58))).is_err());
        let parameter = Param::new(0..=64).unwrap();
        let arms = (0..=64)
            .map(|value| (value..=value, Variant::new("future.algorithm").unwrap()))
            .collect();
        assert!(VariantRule::new(&parameter, arms).is_err());
    }
}
