use std::{error::Error, fmt};

/// Engine-owned attribution, preserved through symbolic graph replay.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileScope {
    site: String,
    layer_type: String,
    layer: Option<u32>,
}

/// A diagnostic name is empty, too long, or contains a line separator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProfileScopeError;

impl fmt::Display for ProfileScopeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .write_str("profile scope names must be nonempty, single-line and at most 256 bytes")
    }
}
impl Error for ProfileScopeError {}

impl ProfileScope {
    /// Validates engine-owned site/layer names at the recording boundary.
    ///
    /// # Errors
    /// Returns an error for empty, oversized or multiline names.
    pub fn new(
        site: String,
        layer_type: String,
        layer: Option<u32>,
    ) -> Result<Self, ProfileScopeError> {
        if [&site, &layer_type]
            .iter()
            .any(|name| name.is_empty() || name.len() > 256 || name.chars().any(char::is_control))
        {
            return Err(ProfileScopeError);
        }
        Ok(Self {
            site,
            layer_type,
            layer,
        })
    }

    /// Returns the engine's call-site scope.
    #[must_use]
    pub fn site(&self) -> &str {
        &self.site
    }

    /// Returns the engine's architectural layer type.
    #[must_use]
    pub fn layer_type(&self) -> &str {
        &self.layer_type
    }

    /// Returns the engine's layer number, if this is a repeated block.
    #[must_use]
    pub const fn layer(&self) -> Option<u32> {
        self.layer
    }

    /// Renders attribution without interpreting model architecture in the host.
    #[must_use]
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({"site": self.site, "layer_type": self.layer_type, "layer": self.layer})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn diagnostic_names_are_bounded_at_construction() {
        assert!(ProfileScope::new("layer/7".into(), "sparse".into(), Some(7)).is_ok());
        for invalid in [String::new(), "a\nb".into(), "x".repeat(257)] {
            assert!(ProfileScope::new(invalid, "dense".into(), None).is_err());
        }
    }
}
