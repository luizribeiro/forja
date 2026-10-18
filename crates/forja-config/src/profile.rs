use std::{collections::BTreeMap, num::NonZeroU32};

use serde::{Deserialize, Deserializer, Serialize, de};
use sha2::{Digest, Sha256};

/// Prevents duplicate-checking work from growing with untrusted input; 64 exceeds the intended
/// number of independently named tuning sets.
const MAX_DEFAULT_TUNINGS: usize = 64;
/// Prevents unbounded build-feature allocation; 64 is well above the feature count of supported
/// engine builds.
const MAX_BUILD_FEATURES: usize = 64;
/// Bounds lookup and hashing work for fixed selections; 1,024 covers every operation site in the
/// current engine families with substantial headroom.
const MAX_FIXED_PICKS: usize = 1_024;
/// Bounds proof work for parameterized selections; 1,024 matches the fixed-selection budget and
/// prevents a profile from multiplying validation cost.
const MAX_VARIANT_RULES: usize = 1_024;
/// Bounds alternatives examined for one tuning; 64 permits broad experiments without allowing a
/// single tuning to dominate validation.
const MAX_TUNING_VARIANTS: usize = 64;
/// Prevents identifier-driven allocation and diagnostic growth; 64 bytes accommodates readable
/// model, tuning, and operation names.
const MAX_NAME_BYTES: usize = 64;

/// A validated engine build profile.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct Profile(ProfileData);

impl Profile {
    /// Returns the profile schema version.
    #[must_use]
    pub const fn schema_version(&self) -> u32 {
        self.0.schema_version
    }

    /// Returns the engine family name.
    #[must_use]
    pub fn family(&self) -> &str {
        &self.0.family
    }

    /// Returns the model identity.
    #[must_use]
    pub const fn model(&self) -> &Model {
        &self.0.model
    }

    /// Returns the build target.
    #[must_use]
    pub const fn target(&self) -> &Target {
        &self.0.target
    }

    /// Returns the component build settings.
    #[must_use]
    pub const fn build(&self) -> &Build {
        &self.0.build
    }

    /// Returns the optional profile-built chat format.
    #[must_use]
    pub const fn chat(&self) -> Option<&Chat> {
        self.0.chat.as_ref()
    }

    /// Returns the numeric formats.
    #[must_use]
    pub const fn numerics(&self) -> &Numerics {
        &self.0.numerics
    }

    /// Returns the supported workload envelope.
    #[must_use]
    pub const fn workload(&self) -> &Workload {
        &self.0.workload
    }

    /// Returns the default tuning names.
    #[must_use]
    pub fn default_tunings(&self) -> &[String] {
        &self.0.default_tunings
    }

    /// Returns the base variant selections.
    #[must_use]
    pub const fn variants(&self) -> &Variants {
        &self.0.variants
    }

    /// Returns tuning-specific variant selections.
    #[must_use]
    pub const fn tuning_variants(&self) -> &BTreeMap<String, Variants> {
        &self.0.tuning_variants
    }

    /// Returns quantization settings for a quantized profile.
    #[must_use]
    pub const fn quantization(&self) -> Option<&Quantization> {
        self.0.quantization.as_ref()
    }

    /// Serializes the profile in deterministic TOML field and map order.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed profile cannot be represented as TOML.
    pub fn canonical_toml(&self) -> Result<String, toml::ser::Error> {
        toml::to_string(self)
    }

    /// Returns the SHA-256 identity of the canonical profile.
    ///
    /// # Errors
    ///
    /// Returns an error if the typed profile cannot be represented as TOML.
    pub fn sha256(&self) -> Result<String, toml::ser::Error> {
        let canonical = self.canonical_toml()?;
        Ok(format!("sha256:{:x}", Sha256::digest(canonical)))
    }
}

impl<'de> Deserialize<'de> for Profile {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let data = ProfileData::deserialize(deserializer)?;
        validate(&data).map_err(de::Error::custom)?;
        Ok(Self(data))
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
struct ProfileData {
    schema_version: u32,
    family: String,
    default_tunings: Vec<String>,
    model: Model,
    target: Target,
    build: Build,
    chat: Option<Chat>,
    numerics: Numerics,
    workload: Workload,
    variants: Variants,
    tuning_variants: BTreeMap<String, Variants>,
    quantization: Option<Quantization>,
}

/// Component build settings declared by a profile.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Build {
    /// Cargo features enabled for the engine component.
    pub features: Vec<String>,
}

/// Profile-built chat-format settings.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Chat {
    /// Tool-call protocol implemented by the component.
    pub tool_format: Option<String>,
    /// Named compatibility quirks understood by the selected format.
    pub tool_format_options: BTreeMap<String, toml::Value>,
    /// Custom chat-format component crate.
    pub component: Option<String>,
}

/// Model identity pinned by a profile.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Model {
    /// Hugging Face repository identifier.
    pub id: String,
    /// Pinned repository commit.
    pub revision: String,
    /// Strong hash of all model weight bytes.
    pub weights_sha256: String,
    /// Strong hash of the model's chat template, when profile-built chat is enabled.
    pub chat_template_sha256: Option<String>,
    /// Strong hash of the model's generation configuration, when profile-built chat is enabled.
    pub generation_config_sha256: Option<String>,
}

/// Component and backend target pinned by a profile.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Target {
    /// WebAssembly component target.
    pub component: ComponentTarget,
    /// Trusted compute backend.
    pub backend: ProfileBackend,
    /// Device family for which selections were made.
    pub device_family: DeviceFamily,
}

/// Supported component target.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ComponentTarget {
    /// WASI Preview 2 component target.
    #[default]
    Wasm32Wasip2,
}

/// Supported profile backend.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ProfileBackend {
    /// Metal GPU backend.
    #[default]
    Metal,
    /// Portable CPU backend.
    Cpu,
}

/// Validated backend device-family slug.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub struct DeviceFamily(String);

impl Default for DeviceFamily {
    fn default() -> Self {
        Self("apple-m3-ultra".to_owned())
    }
}

impl DeviceFamily {
    /// Returns the device-family slug.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Activation and weight formats.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Numerics {
    /// Scalar activation format.
    pub activations: ActivationFormat,
    /// Scalar or quantized weight format.
    pub weights: WeightFormat,
    /// Key/value cache storage format.
    pub kv_cache: KvCacheFormat,
}

/// Supported key/value cache formats.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum KvCacheFormat {
    /// IEEE 754 binary16.
    F16,
    /// Brain floating point.
    Bf16,
    /// IEEE 754 binary32.
    #[default]
    F32,
}

/// Supported activation format.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum ActivationFormat {
    /// IEEE 754 binary32.
    #[default]
    F32,
    /// Brain floating point.
    Bf16,
}

/// Supported weight format.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WeightFormat {
    /// IEEE 754 binary32.
    #[default]
    F32,
    /// Brain floating point.
    Bf16,
    /// Four-bit quantized weights.
    Q4,
}

/// Supported inference workload envelope.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Workload {
    /// Workload contract.
    pub kind: WorkloadKind,
    /// Largest supported batch.
    pub batch: NonZeroU32,
    /// Largest supported context.
    pub max_context: NonZeroU32,
    /// Prefill chunk size.
    pub prefill_chunk: NonZeroU32,
}

impl Default for Workload {
    fn default() -> Self {
        Self {
            kind: WorkloadKind::default(),
            batch: NonZeroU32::MIN,
            max_context: NonZeroU32::MIN,
            prefill_chunk: NonZeroU32::MIN,
        }
    }
}

/// Supported workload kind.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkloadKind {
    /// Autoregressive causal language modeling.
    #[default]
    CausalLm,
}

/// Fixed and parameter-dependent kernel variant selections.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Variants {
    /// Fixed variants keyed by engine-owned dispatch site.
    pub fixed: BTreeMap<String, String>,
    /// Parameter-dependent variants keyed by engine-owned dispatch site.
    pub rules: BTreeMap<String, VariantRule>,
}

/// A piecewise variant selection over one replay parameter.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VariantRule {
    /// Replay parameter name.
    pub parameter: String,
    /// Complete ordered arms.
    pub arms: Vec<VariantArm>,
}

/// One inclusive interval in a variant rule.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct VariantArm {
    /// Inclusive lower bound.
    pub lo: u32,
    /// Inclusive upper bound.
    pub hi: u32,
    /// Backend variant name.
    pub name: String,
}

/// Quantized weight representation.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields, rename_all = "kebab-case")]
pub struct Quantization {
    /// Quantization scheme.
    pub scheme: QuantizationScheme,
    /// Default bit width.
    pub default_bits: u8,
    /// Number of scalar weights in each quantization group.
    pub group_size: NonZeroU32,
    /// Per-weight-path bit-width overrides.
    pub overrides: BTreeMap<String, QuantizationOverride>,
}

impl Default for Quantization {
    fn default() -> Self {
        Self {
            scheme: QuantizationScheme::default(),
            default_bits: 4,
            group_size: NonZeroU32::MIN,
            overrides: BTreeMap::new(),
        }
    }
}

/// Supported quantization scheme.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum QuantizationScheme {
    /// MLX affine group quantization.
    #[default]
    MlxAffine,
}

/// Quantization settings for one engine-owned weight path.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct QuantizationOverride {
    /// Bit width for the matched weight path.
    pub bits: u8,
}

fn validate(profile: &ProfileData) -> Result<(), String> {
    if profile.schema_version != 1 {
        return Err(format!(
            "unsupported profile schema version {}",
            profile.schema_version
        ));
    }
    require_name("family", &profile.family)?;
    require_name("model.id", &profile.model.id)?;
    validate_model_id(&profile.model.id)?;
    validate_hex("model.revision", &profile.model.revision, 40, "")?;
    validate_hex(
        "model.weights-sha256",
        &profile.model.weights_sha256,
        64,
        "sha256:",
    )?;
    if let Some(hash) = &profile.model.chat_template_sha256 {
        validate_hex("model.chat-template-sha256", hash, 64, "sha256:")?;
    }
    if let Some(hash) = &profile.model.generation_config_sha256 {
        validate_hex("model.generation-config-sha256", hash, 64, "sha256:")?;
    }
    validate_slug(
        "target.device-family",
        profile.target.device_family.as_str(),
    )?;
    validate_build_features(&profile.build.features)?;
    if let Some(chat) = &profile.chat {
        match (&chat.tool_format, &chat.component) {
            (Some(format), None) => {
                validate_slug("chat.tool-format", format)?;
                if profile.model.chat_template_sha256.is_none()
                    || profile.model.generation_config_sha256.is_none()
                {
                    return Err(
                        "chat.tool-format requires model chat-template-sha256 and generation-config-sha256"
                            .to_owned(),
                    );
                }
            }
            (None, Some(component)) => {
                validate_slug("chat.component", component)?;
                if !chat.tool_format_options.is_empty() {
                    return Err("chat.tool-format-options requires chat.tool-format".to_owned());
                }
            }
            (Some(_), Some(_)) => {
                return Err("chat must select either tool-format or component".to_owned());
            }
            (None, None) => return Err("chat must select tool-format or component".to_owned()),
        }
        require_count(
            "chat.tool-format-options",
            chat.tool_format_options.len(),
            MAX_DEFAULT_TUNINGS,
        )?;
    }
    if profile.workload.prefill_chunk > profile.workload.max_context {
        return Err("workload.prefill-chunk must not exceed max-context".to_owned());
    }
    validate_tunings(&profile.default_tunings)?;
    validate_variants(&profile.variants, profile.workload.max_context)?;
    require_count(
        "tuning-variants",
        profile.tuning_variants.len(),
        MAX_TUNING_VARIANTS,
    )?;
    for (tuning, variants) in &profile.tuning_variants {
        require_name("tuning-variants name", tuning)?;
        validate_variants(variants, profile.workload.max_context)?;
    }
    validate_quantization(profile)
}

fn validate_build_features(features: &[String]) -> Result<(), String> {
    require_count("build.features", features.len(), MAX_BUILD_FEATURES)?;
    let mut sorted = features.to_vec();
    sorted.sort();
    for feature in &sorted {
        require_name("build feature", feature)?;
    }
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("build.features must be unique".to_owned());
    }
    Ok(())
}

fn validate_slug(field: &str, value: &str) -> Result<(), String> {
    require_name(field, value)?;
    if value.split('-').any(|part| {
        part.is_empty()
            || !part
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit())
    }) {
        return Err(format!(
            "{field} must contain lowercase ASCII letters and digits separated by single hyphens"
        ));
    }
    Ok(())
}

fn require_name(field: &str, value: &str) -> Result<(), String> {
    if value.is_empty() {
        Err(format!("{field} must not be empty"))
    } else if value.len() > MAX_NAME_BYTES {
        Err(format!(
            "{field} exceeds the {MAX_NAME_BYTES}-byte name limit"
        ))
    } else {
        Ok(())
    }
}

fn require_count(field: &str, count: usize, maximum: usize) -> Result<(), String> {
    if count > maximum {
        Err(format!("{field} exceeds the {maximum}-entry limit"))
    } else {
        Ok(())
    }
}

fn validate_model_id(id: &str) -> Result<(), String> {
    if id.starts_with('/')
        || id
            .split('/')
            .any(|segment| segment.is_empty() || matches!(segment, "." | ".."))
    {
        return Err("model.id must contain only nonempty relative segments".to_owned());
    }
    Ok(())
}

fn validate_hex(field: &str, value: &str, digits: usize, prefix: &str) -> Result<(), String> {
    let Some(hex) = value.strip_prefix(prefix) else {
        return Err(format!("{field} must start with {prefix:?}"));
    };
    if hex.len() != digits
        || !hex
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(format!(
            "{field} must contain {digits} lowercase hexadecimal digits"
        ));
    }
    Ok(())
}

fn validate_tunings(tunings: &[String]) -> Result<(), String> {
    require_count("default-tunings", tunings.len(), MAX_DEFAULT_TUNINGS)?;
    let mut sorted = tunings.to_vec();
    sorted.sort();
    for tuning in &sorted {
        require_name("default tuning", tuning)?;
    }
    if sorted.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("default-tunings must be unique".to_owned());
    }
    Ok(())
}

fn validate_variants(variants: &Variants, max_context: NonZeroU32) -> Result<(), String> {
    require_count("fixed variant picks", variants.fixed.len(), MAX_FIXED_PICKS)?;
    require_count("variant rules", variants.rules.len(), MAX_VARIANT_RULES)?;
    for (site, name) in &variants.fixed {
        require_name("variant site", site)?;
        require_name("variant name", name)?;
        if variants.rules.contains_key(site) {
            return Err(format!("variant site {site:?} appears in fixed and rules"));
        }
    }
    for (site, rule) in &variants.rules {
        require_name("variant site", site)?;
        if rule.parameter != "position" {
            return Err(format!(
                "variant rule {site:?} has unknown parameter {:?}",
                rule.parameter
            ));
        }
        validate_arms(site, &rule.arms, max_context.get() - 1)?;
    }
    Ok(())
}

fn validate_arms(site: &str, arms: &[VariantArm], final_hi: u32) -> Result<(), String> {
    let mut next_lo = 0_u64;
    for arm in arms {
        require_name("variant name", &arm.name)?;
        if u64::from(arm.lo) != next_lo || arm.lo > arm.hi {
            return Err(format!(
                "variant rule {site:?} arms must be ordered, complete, and non-overlapping"
            ));
        }
        next_lo = u64::from(arm.hi) + 1;
    }
    if next_lo != u64::from(final_hi) + 1 {
        return Err(format!(
            "variant rule {site:?} arms must cover 0 through {final_hi}"
        ));
    }
    Ok(())
}

fn validate_quantization(profile: &ProfileData) -> Result<(), String> {
    match (profile.numerics.weights, &profile.quantization) {
        (WeightFormat::Q4, Some(quantization)) => {
            if quantization.default_bits != 4 {
                return Err("quantization.default-bits must match q4 weights".to_owned());
            }
            for (path, value) in &quantization.overrides {
                require_name("quantization override path", path)?;
                if value.bits == 0 || value.bits > 8 {
                    return Err(format!(
                        "quantization override {path:?} bits must be in 1..=8"
                    ));
                }
            }
            Ok(())
        }
        (WeightFormat::Q4, None) => Err("q4 weights require [quantization]".to_owned()),
        (_, Some(_)) => Err("dense profiles must not contain [quantization]".to_owned()),
        (_, None) => Ok(()),
    }
}
