use std::{error::Error, time::Duration};

use serde_json::{Value, json};

pub(super) fn capture(stage: &str) -> Result<Value, Box<dyn Error>> {
    if cfg!(target_os = "macos") {
        Ok(serde_json::to_value(crate::machine_load::capture(stage)?)?)
    } else {
        Ok(Value::Null)
    }
}

pub(super) fn provenance(
    device: &str,
    backend: forja_config::BackendKind,
    before: &Value,
    after: &Value,
    elapsed: Duration,
) -> Result<Value, Box<dyn Error>> {
    use crate::benchmark::command_output as command;
    let macos = cfg!(target_os = "macos");
    let version = command(
        if macos { "sw_vers" } else { "uname" },
        if macos { &["-productVersion"] } else { &["-r"] },
    )?;
    let build = command(
        if macos { "sw_vers" } else { "uname" },
        if macos { &["-buildVersion"] } else { &["-v"] },
    )?;
    let gpu_cores = if macos && backend == forja_config::BackendKind::Metal {
        Some(crate::benchmark::gpu_core_count()?)
    } else {
        None
    };
    let memory = if macos {
        Some(command("sysctl", &["-n", "hw.memsize"])?.parse::<u64>()?)
    } else {
        None
    };
    Ok(
        json!({"commit":crate::provenance::commit(), "dirty":crate::provenance::dirty(), "clean_source":!crate::provenance::dirty(),
        "binary_sha256":golden_fixtures::sha256_file(std::env::current_exe()?)?,
        "build_environment":crate::provenance::BUILD_ENVIRONMENT,
        "device":device, "gpu_cores":gpu_cores, "memory_bytes":memory,
        "machine":command("hostname", &[])?, "architecture":std::env::consts::ARCH,
        "os":std::env::consts::OS, "os_version":version, "os_build":build,
        "macos":if macos {Some(&version)} else {None},
        "date":command("date", &["-u", "+%Y-%m-%dT%H:%M:%SZ"])?, "duration_s":elapsed.as_secs_f64(),
        "machine_load":{"before":before,"after":after}}),
    )
}
