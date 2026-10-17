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
    use std::{env, fs, path::PathBuf};

    use forja_chat_format::{ChatTemplate, ToolFormat};
    use qwen3_coder_xml::{Options, Qwen3CoderXml};

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

    #[test]
    fn renders_existing_prompt_goldens() {
        let goldens: serde_json::Value = serde_json::from_str(include_str!(
            "../../../chat-formats/qwen3-coder-xml/tests/fixtures/chat-template.json"
        ))
        .unwrap();
        let template = ChatTemplate::compile(data::TEMPLATE.to_owned()).unwrap();
        for golden in goldens.as_array().unwrap() {
            let messages = serde_json::to_string(&golden["messages"]).unwrap();
            let actual = Qwen3CoderXml::render(
                &template,
                &messages,
                "null",
                r#"{"add_generation_prompt":true}"#,
                &Options,
            )
            .unwrap();
            assert_eq!(actual, golden["rendered"].as_str().unwrap());
        }
    }

    #[test]
    #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
    fn embedded_data_matches_the_pinned_model() {
        let root = PathBuf::from(env::var_os("FORJA_MODELS").expect("FORJA_MODELS is not set"))
            .join("Qwen3-Coder-30B-A3B-Instruct-4bit");
        assert_eq!(
            fs::read_to_string(root.join("chat_template.jinja")).unwrap(),
            data::TEMPLATE
        );
        let config: serde_json::Value =
            serde_json::from_slice(&fs::read(root.join("generation_config.json")).unwrap())
                .unwrap();
        assert_eq!(
            config["eos_token_id"],
            serde_json::json!(data::STOP_TOKEN_IDS)
        );
        assert_eq!(config["temperature"], data::TEMPERATURE.unwrap());
        assert_eq!(config["top_p"], data::TOP_P.unwrap());
        assert_eq!(config["top_k"], data::TOP_K.unwrap());
        assert_eq!(
            config["repetition_penalty"],
            data::REPETITION_PENALTY.unwrap()
        );
    }
}
