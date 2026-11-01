//! Pinned text-only Qwen3.8 chat format with upstream thinking controls.
#![cfg_attr(not(target_family = "wasm"), forbid(unsafe_code))]
use forja_chat_format::{ChatTemplate, Event, IncrementalParser, ToolFormat};
use serde_json::Value;
#[cfg(any(target_family = "wasm", test))]
mod template;
/// Text chat with the pinned thinking and history controls.
pub struct QwenChat;
/// Plain streamed assistant text.
#[derive(Default)]
pub struct TextParser;
impl IncrementalParser for TextParser {
    fn parse(&mut self, chunk: &str) -> Result<Vec<Event>, String> {
        Ok(vec![Event::TextDelta(chunk.to_owned())])
    }
}
impl ToolFormat for QwenChat {
    type Options = ();
    type Parser = TextParser;
    fn options(options: &str) -> Result<(), String> {
        if serde_json::from_str::<Value>(options).map_err(|e| e.to_string())?
            != serde_json::json!({})
        {
            return Err("qwen38 requires empty format options".to_owned());
        }
        Ok(())
    }
    fn trigger((): &()) -> &'static str {
        ""
    }
    fn normalize_messages(messages: &mut Value, (): &()) -> Result<(), String> {
        let items = messages.as_array().ok_or("messages must be an array")?;
        for message in items {
            if let Some(content) = message.get("content") {
                let text = content.is_string()
                    || content.is_null()
                    || content.as_array().is_some_and(|parts| {
                        parts.iter().all(|part| {
                            part.get("text").is_some_and(Value::is_string)
                                && part.get("type").is_none_or(|kind| kind == "text")
                                && !["image", "image_url", "video"]
                                    .iter()
                                    .any(|key| part.get(key).is_some())
                        })
                    });
                if !text {
                    return Err("only text content is supported".to_owned());
                }
            }
        }
        qwen3_coder_xml::Qwen3CoderXml::normalize_messages(messages, &qwen3_coder_xml::Options)?;
        Ok(())
    }
    fn render(
        template: &ChatTemplate,
        messages: &str,
        tools: &str,
        request: &str,
        (): &(),
    ) -> Result<String, String> {
        let tools: Value = serde_json::from_str(tools).map_err(|e| e.to_string())?;
        if !tools.is_null() && !tools.is_array() {
            return Err("tools must be an array".to_owned());
        }
        let mut messages: Value = serde_json::from_str(messages).map_err(|e| e.to_string())?;
        Self::normalize_messages(&mut messages, &())?;
        let request: Value = serde_json::from_str(request).map_err(|e| e.to_string())?;
        request
            .as_object()
            .ok_or("request options must be an object")?;
        template.render(
            &messages.to_string(),
            &tools.to_string(),
            &request.to_string(),
        )
    }
    fn grammar(_: &str, _: &str, _: &str, (): &()) -> Result<String, String> {
        Err("constrained tools are unsupported".to_owned())
    }
}
#[cfg(target_family = "wasm")]
mod data {
    pub const TEMPLATE: &str = super::template::TEMPLATE;
    pub const TOOL_FORMAT_OPTIONS: &str = "{}";
    pub const STOP_TOKEN_IDS: &[u32] = &[248_046, 248_044];
    pub const TEMPERATURE: Option<f32> = Some(0.0);
    pub const TOP_P: Option<f32> = None;
    pub const TOP_K: Option<u32> = None;
    pub const REPETITION_PENALTY: Option<f32> = None;
    pub const MAX_NEW_TOKENS: Option<u32> = None;
}
#[cfg(target_family = "wasm")]
forja_chat_format::export_component!(crate::QwenChat, data);
#[cfg(test)]
mod tests {
    use super::*;
    fn render(messages: &str, options: &str) -> Result<String, String> {
        QwenChat::render(
            &ChatTemplate::compile(super::template::TEMPLATE.to_owned())
                .map_err(|error| error.to_string())?,
            messages,
            "null",
            options,
            &(),
        )
    }
    #[test]
    fn thinking_and_history_match_pinned_transformers_goldens() {
        let fixtures: Value =
            serde_json::from_str(include_str!("../tests/fixtures/render.json")).unwrap();
        for fixture in fixtures.as_array().unwrap() {
            assert_eq!(
                QwenChat::render(
                    &ChatTemplate::compile(super::template::TEMPLATE.to_owned()).unwrap(),
                    &fixture["messages"].to_string(),
                    &fixture["tools"].to_string(),
                    &fixture["options"].to_string(),
                    &(),
                )
                .unwrap(),
                fixture["rendered"].as_str().unwrap(),
                "{}",
                fixture["name"]
            );
        }
    }
    #[test]
    fn refuses_assistant_only_history() {
        assert!(
            render(
                r#"[{"role":"assistant","content":"","tool_calls":[]}]"#,
                "{}"
            )
            .is_err()
        );
    }
    #[test]
    fn refuses_tool_only_history() {
        assert!(render(r#"[{"role":"tool","content":"result"}]"#, "{}").is_err());
    }
    #[test]
    fn renders_non_thinking_and_refuses_unsupported_requests() {
        assert_eq!(
            render(
                r#"[{"role":"user","content":"Hello"}]"#,
                r#"{"add_generation_prompt":true,"enable_thinking":false}"#
            )
            .unwrap(),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert!(
            render(
                r#"[{"role":"user","content":[{"type":"image","image":"x"}]}]"#,
                "{}"
            )
            .is_err()
        );
        assert!(render("[]", r#"{"enable_thinking":true}"#).is_err());
        assert!(QwenChat::options(r#"{"unknown":true}"#).is_err());
        assert_eq!(
            TextParser.parse("hello").unwrap(),
            [Event::TextDelta("hello".to_owned())]
        );
    }
    #[test]
    fn rejects_video_and_non_object_encoded_tool_arguments() {
        for content in [r#"[{"type":"video","video":"x"}]"#, r#"{"text":"x"}"#] {
            assert!(
                render(
                    &format!("[{{\"role\":\"user\",\"content\":{content}}}]"),
                    "{}"
                )
                .is_err()
            );
        }
        assert!(render(r#"[{"role":"user","content":"hi"},{"role":"assistant","content":null,"tool_calls":[{"function":{"name":"x","arguments":"[]"}}]}]"#, "{}").is_err());
    }
}
