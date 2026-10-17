//! Generates model-owned data for the generic chat-format component.

use std::{env, fs, io, path::PathBuf};

use serde_json::Value;

mod build_support;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest = PathBuf::from(required("CARGO_MANIFEST_DIR")?);
    let fixtures = manifest.join("../../chat-formats/qwen3-coder-xml/tests/fixtures");
    let test_fixtures = env::var_os("CARGO_FEATURE_TEST_FIXTURES").is_some();
    let template = build_support::source(
        "FORJA_CHAT_TEMPLATE",
        env::var_os("FORJA_CHAT_TEMPLATE"),
        &fixtures.join("chat_template.jinja"),
        test_fixtures,
    )?;
    let generation = build_support::source(
        "FORJA_GENERATION_CONFIG",
        env::var_os("FORJA_GENERATION_CONFIG"),
        &fixtures.join("generation_config.json"),
        test_fixtures,
    )?;
    let config: Value = serde_json::from_slice(&fs::read(generation)?)?;
    let options = env::var("FORJA_TOOL_FORMAT_OPTIONS").unwrap_or_else(|_| "{}".to_owned());
    let stop_ids = config["eos_token_id"]
        .as_array()
        .ok_or("generation config has no eos_token_id array")?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| u32::try_from(value).ok())
                .ok_or("eos_token_id is not a u32")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let generated = format!(
        "pub const TEMPLATE: &str = {template:?};\n\
         pub const TOOL_FORMAT_OPTIONS: &str = {options:?};\n\
         #[allow(clippy::unreadable_literal)]\n\
         pub const STOP_TOKEN_IDS: &[u32] = &{stop_ids:?};\n\
         pub const TEMPERATURE: Option<f32> = {};\n\
         pub const TOP_P: Option<f32> = {};\n\
         pub const TOP_K: Option<u32> = {};\n\
         pub const REPETITION_PENALTY: Option<f32> = {};\n\
         pub const MAX_NEW_TOKENS: Option<u32> = {};\n",
        float(&config, "temperature")?,
        float(&config, "top_p")?,
        integer(&config, "top_k")?,
        float(&config, "repetition_penalty")?,
        integer(&config, "max_new_tokens")?,
        template = fs::read_to_string(template)?,
    );
    fs::write(
        PathBuf::from(required("OUT_DIR")?).join("chat_data.rs"),
        generated,
    )?;
    println!("cargo::rerun-if-env-changed=FORJA_CHAT_TEMPLATE");
    println!("cargo::rerun-if-env-changed=FORJA_GENERATION_CONFIG");
    println!("cargo::rerun-if-env-changed=FORJA_TOOL_FORMAT_OPTIONS");
    Ok(())
}

fn float(config: &Value, field: &str) -> Result<String, String> {
    optional_number(config, field, "f32")
}

fn integer(config: &Value, field: &str) -> Result<String, String> {
    optional_number(config, field, "u32")
}

fn optional_number(config: &Value, field: &str, suffix: &str) -> Result<String, String> {
    match config.get(field) {
        None | Some(Value::Null) => Ok("None".to_owned()),
        Some(value) if value.is_number() => Ok(format!("Some({value}_{suffix})")),
        Some(_) => Err(format!("generation config {field} is not numeric")),
    }
}

fn required(name: &str) -> Result<std::ffi::OsString, io::Error> {
    env::var_os(name).ok_or_else(|| io::Error::other(format!("{name} is not set")))
}
