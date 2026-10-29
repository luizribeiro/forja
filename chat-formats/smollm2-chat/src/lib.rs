//! Pinned text-only `ChatML` format for `SmolLM2-135M-Instruct`.

#![cfg_attr(not(target_family = "wasm"), forbid(unsafe_code))]

#[cfg(any(target_family = "wasm", test))]
mod template;

use forja_chat_format::{Event, IncrementalParser, ToolFormat};

/// Plain assistant text; this model does not declare a tool-call protocol.
pub struct SmolChat;

/// Stateless UTF-8 text parser.
#[derive(Default)]
pub struct TextParser;

impl IncrementalParser for TextParser {
    fn parse(&mut self, chunk: &str) -> Result<Vec<Event>, String> {
        Ok(vec![Event::TextDelta(chunk.to_owned())])
    }
}

impl ToolFormat for SmolChat {
    type Options = ();
    type Parser = TextParser;

    fn options(options: &str) -> Result<(), String> {
        if serde_json::from_str::<serde_json::Value>(options).map_err(|error| error.to_string())?
            == serde_json::json!({})
        {
            Ok(())
        } else {
            Err("SmolLM2 has no chat-format options".to_owned())
        }
    }

    fn trigger((): &()) -> &'static str {
        ""
    }

    fn grammar(_: &str, _: &str, _: &str, (): &()) -> Result<String, String> {
        Err("SmolLM2-135M-Instruct does not support constrained tool calls".to_owned())
    }
}

#[cfg(target_family = "wasm")]
mod data {
    pub use super::template::TEMPLATE;
    pub const TOOL_FORMAT_OPTIONS: &str = "{}";
    pub const STOP_TOKEN_IDS: &[u32] = &[2];
    pub const TEMPERATURE: Option<f32> = None;
    pub const TOP_P: Option<f32> = None;
    pub const TOP_K: Option<u32> = None;
    pub const REPETITION_PENALTY: Option<f32> = None;
    pub const MAX_NEW_TOKENS: Option<u32> = None;
}

#[cfg(target_family = "wasm")]
forja_chat_format::export_component!(crate::SmolChat, data);

#[cfg(test)]
mod tests {
    use super::*;
    use forja_chat_format::ChatTemplate;

    #[test]
    fn renders_the_pinned_template_and_preserves_streamed_text() {
        let template = ChatTemplate::compile(template::TEMPLATE.to_owned()).unwrap();
        let actual = SmolChat::render(
            &template,
            r#"[{"role":"user","content":"Hello"}]"#,
            "null",
            r#"{"add_generation_prompt":true}"#,
            &(),
        )
        .unwrap();
        assert_eq!(
            actual,
            "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"
        );
        assert_eq!(
            TextParser.parse("hello").unwrap(),
            [Event::TextDelta("hello".to_owned())]
        );
        assert!(SmolChat::options(r#"{"unknown":true}"#).is_err());
        assert!(SmolChat::grammar("[]", "null", "null", &()).is_err());
    }
}
