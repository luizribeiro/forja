//! Generic profile-built chat-format component.

#![cfg_attr(not(target_family = "wasm"), forbid(unsafe_code))]

#[cfg_attr(not(target_family = "wasm"), allow(dead_code))]
mod data {
    include!(concat!(env!("OUT_DIR"), "/chat_data.rs"));
}

#[cfg(all(target_family = "wasm", feature = "qwen3-coder-xml"))]
forja_chat_format::export_component!(qwen3_coder_xml::Qwen3CoderXml, data);

#[cfg(test)]
mod tests {
    use super::data;

    #[test]
    fn embeds_model_chat_data() {
        assert!(data::TEMPLATE.contains("<tool_call>"));
        assert_eq!(data::STOP_TOKEN_IDS, [151_645, 151_643]);
        assert_eq!(data::TEMPERATURE, Some(0.7));
        assert_eq!(data::TOP_P, Some(0.8));
        assert_eq!(data::TOP_K, Some(20));
        assert_eq!(data::REPETITION_PENALTY, Some(1.05));
        assert_eq!(data::MAX_NEW_TOKENS, None);
        assert_eq!(data::TOOL_FORMAT_OPTIONS, "{}");
    }
}
