use serde_json::{Value, json};
use sha2::{Digest, Sha256};
#[cfg(target_os = "macos")]
use std::error::Error;

#[cfg(target_os = "macos")]
pub(crate) fn identity() -> Result<Value, Box<dyn Error>> {
    use forja_core::Backend;
    let backend = forja_metal::MetalBackend::new()?;
    let device = backend.device_info()?;
    let displays: Value = serde_json::from_str(&crate::benchmark::command_output(
        "system_profiler",
        &["SPDisplaysDataType", "-json"],
    )?)?;
    let gpu = displays["SPDisplaysDataType"]
        .as_array()
        .ok_or("missing GPU identity")?
        .iter()
        .find(|gpu| gpu["sppci_model"].as_str() == Some(backend.device_name().as_str()))
        .ok_or("selected GPU absent from system profiler")?;
    let cores = gpu["sppci_cores"]
        .as_str()
        .ok_or("missing GPU core count")?
        .parse::<u32>()?;
    Ok(
        json!({"name": backend.device_name(), "family": device.family,
        "core_count": cores, "memory_bytes": forja_metal::MetalBackend::physical_memory_size()?,
        "os_version": crate::benchmark::command_output("sw_vers", &["-productVersion"])?,
        "os_build": crate::benchmark::command_output("sw_vers", &["-buildVersion"])?}),
    )
}

pub(crate) fn cache_key(identity: &Value) -> String {
    format!(
        "{:x}",
        Sha256::digest(
            json!({"calibrator_schema": 2, "identity": identity})
                .to_string()
                .as_bytes()
        )
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn identity_changes_invalidate_cache_keys() {
        let identity = json!({"name":"test-device", "family":"test", "core_count": 7, "memory_bytes": 33, "os_version":"one", "os_build":"build-one"});
        for (key, replacement) in [
            ("name", json!("other-device")),
            ("os_build", json!("build-two")),
            ("family", json!("other")),
            ("core_count", json!(8)),
            ("memory_bytes", json!(34)),
            ("os_version", json!("two")),
        ] {
            let mut changed = identity.clone();
            changed[key] = replacement;
            assert_ne!(cache_key(&identity), cache_key(&changed));
        }
    }
}
