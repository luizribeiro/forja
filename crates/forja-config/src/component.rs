use std::{error::Error, fmt, str};

use wasmparser::{Encoding, Parser, Payload};

use crate::Profile;

const PROFILE_SECTION: &str = "forja.profile.v1";
const PROFILE_SECTION_PREFIX: &str = "forja.profile.";
const MAX_PROFILE_BYTES: usize = 1024 * 1024;

/// A failure to encode or read an embedded engine profile.
#[derive(Debug, Eq, PartialEq)]
pub enum ProfileSectionError {
    /// The bytes are not a well-formed WebAssembly component.
    InvalidComponent(String),
    /// The component has no supported profile section.
    Missing,
    /// The component has more than one supported profile section.
    Duplicate,
    /// The section does not contain canonical valid profile TOML.
    Malformed(String),
    /// The component contains a profile section version this host does not support.
    UnknownVersion(String),
}

impl fmt::Display for ProfileSectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidComponent(error) => write!(formatter, "invalid component: {error}"),
            Self::Missing => formatter.write_str("component profile section is missing"),
            Self::Duplicate => formatter.write_str("component profile section is duplicated"),
            Self::Malformed(error) => write!(formatter, "malformed component profile: {error}"),
            Self::UnknownVersion(name) => {
                write!(formatter, "unknown component profile section {name:?}")
            }
        }
    }
}

impl Error for ProfileSectionError {}

/// Appends a canonical engine profile custom section to a component.
///
/// # Errors
///
/// Returns an error when the input is not a component, already has a profile section, the profile
/// cannot be serialized, or section sizes exceed WebAssembly limits.
pub fn embed_profile(component: &[u8], profile: &Profile) -> Result<Vec<u8>, ProfileSectionError> {
    if scan_profile(component)?.is_some() {
        return Err(ProfileSectionError::Duplicate);
    }
    let canonical = profile
        .canonical_toml()
        .map_err(|error| ProfileSectionError::Malformed(error.to_string()))?;
    append_custom_section(component, PROFILE_SECTION, canonical.as_bytes())
}

/// Reads and validates the canonical engine profile embedded in a component.
///
/// # Errors
///
/// Returns an error when the component or section is invalid, the section is missing or repeated,
/// or its version is unsupported.
pub fn read_profile(component: &[u8]) -> Result<Profile, ProfileSectionError> {
    let bytes = scan_profile(component)?.ok_or(ProfileSectionError::Missing)?;
    let source =
        str::from_utf8(bytes).map_err(|error| ProfileSectionError::Malformed(error.to_string()))?;
    let profile = toml::from_str::<Profile>(source)
        .map_err(|error| ProfileSectionError::Malformed(error.to_string()))?;
    let canonical = profile
        .canonical_toml()
        .map_err(|error| ProfileSectionError::Malformed(error.to_string()))?;
    if source != canonical {
        return Err(ProfileSectionError::Malformed(
            "profile TOML is not canonical".to_owned(),
        ));
    }
    Ok(profile)
}

fn scan_profile(component: &[u8]) -> Result<Option<&[u8]>, ProfileSectionError> {
    let mut profile = None;
    let mut depth = 0_u32;
    let mut root_encoding = None;
    for payload in Parser::new(0).parse_all(component) {
        let payload =
            payload.map_err(|error| ProfileSectionError::InvalidComponent(error.to_string()))?;
        match payload {
            Payload::Version { encoding, .. } if depth == 0 => root_encoding = Some(encoding),
            Payload::CustomSection(section) if depth == 0 => match section.name() {
                PROFILE_SECTION if profile.is_some() => return Err(ProfileSectionError::Duplicate),
                PROFILE_SECTION if section.data().len() > MAX_PROFILE_BYTES => {
                    return Err(ProfileSectionError::Malformed(format!(
                        "profile section exceeds the {MAX_PROFILE_BYTES}-byte limit"
                    )));
                }
                PROFILE_SECTION => profile = Some(section.data()),
                name if name.starts_with(PROFILE_SECTION_PREFIX) => {
                    return Err(ProfileSectionError::UnknownVersion(name.to_owned()));
                }
                _ => {}
            },
            Payload::ModuleSection { .. } | Payload::ComponentSection { .. } => {
                depth = depth.checked_add(1).ok_or_else(|| {
                    ProfileSectionError::InvalidComponent("nesting depth overflows u32".to_owned())
                })?;
            }
            Payload::End(_) if depth > 0 => depth -= 1,
            _ => {}
        }
    }
    if root_encoding != Some(Encoding::Component) {
        return Err(ProfileSectionError::InvalidComponent(
            "expected component encoding".to_owned(),
        ));
    }
    Ok(profile)
}

fn append_custom_section(
    component: &[u8],
    name: &str,
    data: &[u8],
) -> Result<Vec<u8>, ProfileSectionError> {
    let name_length = u32::try_from(name.len())
        .map_err(|_| ProfileSectionError::Malformed("section name is too long".to_owned()))?;
    let name_prefix_length = u32_leb_length(name_length);
    let payload_length = name_prefix_length
        .checked_add(name.len())
        .and_then(|length| length.checked_add(data.len()))
        .and_then(|length| u32::try_from(length).ok())
        .ok_or_else(|| ProfileSectionError::Malformed("profile section is too long".to_owned()))?;
    let payload_capacity = usize::try_from(payload_length)
        .map_err(|_| ProfileSectionError::Malformed("profile section is too long".to_owned()))?;
    let capacity = component
        .len()
        .checked_add(6)
        .and_then(|length| length.checked_add(payload_capacity))
        .ok_or_else(|| ProfileSectionError::Malformed("component is too long".to_owned()))?;
    let mut result = Vec::with_capacity(capacity);
    result.extend_from_slice(component);
    result.push(0);
    push_u32_leb(&mut result, payload_length);
    push_u32_leb(&mut result, name_length);
    result.extend_from_slice(name.as_bytes());
    result.extend_from_slice(data);
    Ok(result)
}

fn u32_leb_length(mut value: u32) -> usize {
    let mut length = 1;
    while value >= 0x80 {
        value >>= 7;
        length += 1;
    }
    length
}

fn push_u32_leb(output: &mut Vec<u8>, mut value: u32) {
    loop {
        let byte = value.to_le_bytes()[0] & 0x7f;
        value >>= 7;
        output.push(if value == 0 { byte } else { byte | 0x80 });
        if value == 0 {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPONENT: &[u8] = b"\0asm\x0d\0\x01\0";
    const PROFILE: &str = r#"
schema-version = 1
family = "test"

[model]
id = "org/model"
revision = "0000000000000000000000000000000000000000"
weights-sha256 = "sha256:0000000000000000000000000000000000000000000000000000000000000000"
"#;

    fn profile() -> Profile {
        toml::from_str(PROFILE).unwrap()
    }

    #[test]
    fn embedding_is_deterministic_and_round_trips() {
        let profile = profile();
        let first = embed_profile(COMPONENT, &profile).unwrap();
        let second = embed_profile(COMPONENT, &profile).unwrap();
        assert_eq!(first, second);
        assert_eq!(read_profile(&first).unwrap(), profile);
        assert_eq!(
            embed_profile(&first, &profile),
            Err(ProfileSectionError::Duplicate)
        );
    }

    #[test]
    fn missing_duplicate_malformed_and_unknown_sections_fail() {
        assert_eq!(read_profile(COMPONENT), Err(ProfileSectionError::Missing));

        let once = append_custom_section(COMPONENT, PROFILE_SECTION, PROFILE.as_bytes()).unwrap();
        let twice = append_custom_section(&once, PROFILE_SECTION, PROFILE.as_bytes()).unwrap();
        assert_eq!(read_profile(&twice), Err(ProfileSectionError::Duplicate));

        let malformed = append_custom_section(COMPONENT, PROFILE_SECTION, b"not toml").unwrap();
        assert!(matches!(
            read_profile(&malformed),
            Err(ProfileSectionError::Malformed(_))
        ));

        let unknown = append_custom_section(COMPONENT, "forja.profile.v2", b"").unwrap();
        assert_eq!(
            read_profile(&unknown),
            Err(ProfileSectionError::UnknownVersion(
                "forja.profile.v2".to_owned()
            ))
        );
    }

    #[test]
    fn bounds_embedded_profile_bytes_before_parsing() {
        let at_limit =
            append_custom_section(COMPONENT, PROFILE_SECTION, &vec![0; MAX_PROFILE_BYTES]).unwrap();
        assert_eq!(
            scan_profile(&at_limit).unwrap().unwrap().len(),
            MAX_PROFILE_BYTES
        );

        let over_limit =
            append_custom_section(COMPONENT, PROFILE_SECTION, &vec![0; MAX_PROFILE_BYTES + 1])
                .unwrap();
        assert!(matches!(
            scan_profile(&over_limit),
            Err(ProfileSectionError::Malformed(message)) if message.contains("byte limit")
        ));
    }
}
