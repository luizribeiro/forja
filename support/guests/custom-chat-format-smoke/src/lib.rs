//! Minimal custom component used to verify the profile escape hatch.

#![cfg_attr(not(target_family = "wasm"), forbid(unsafe_code))]

#[cfg(target_family = "wasm")]
mod data {
    pub const TEMPLATE: &str = "{{ messages[0].content }}";
    pub const TOOL_FORMAT_OPTIONS: &str = "{}";
    pub const STOP_TOKEN_IDS: &[u32] = &[7];
    pub const TEMPERATURE: Option<f32> = None;
    pub const TOP_P: Option<f32> = None;
    pub const TOP_K: Option<u32> = None;
    pub const REPETITION_PENALTY: Option<f32> = None;
    pub const MAX_NEW_TOKENS: Option<u32> = None;
}

#[cfg(target_family = "wasm")]
forja_chat_format::export_component!(qwen3_coder_xml::Qwen3CoderXml, data, "../../../wit");
