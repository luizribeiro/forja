//! Pinned text-only Qwen3.8 chat format with explicit non-thinking generation.
#![cfg_attr(not(target_family = "wasm"), forbid(unsafe_code))]
use forja_chat_format::{ChatTemplate, Event, IncrementalParser, ToolFormat};
use serde_json::Value;
#[cfg(any(target_family = "wasm", test))]
mod template;
/// Non-thinking text chat. Tools and multimodal messages require another format.
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
        let messages = messages.as_array().ok_or("messages must be an array")?;
        for message in messages {
            if !message.get("content").is_some_and(Value::is_string)
                || message.get("tool_calls").is_some()
                || message.get("role").is_some_and(|role| role == "tool")
            {
                return Err("qwen38 currently accepts text chat without tools".to_owned());
            }
        }
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
        if !tools.is_null() && tools != serde_json::json!([]) {
            return Err("tools are unsupported".to_owned());
        }
        let mut messages: Value = serde_json::from_str(messages).map_err(|e| e.to_string())?;
        Self::normalize_messages(&mut messages, &())?;
        let mut request: Value = serde_json::from_str(request).map_err(|e| e.to_string())?;
        let options = request
            .as_object_mut()
            .ok_or("request options must be an object")?;
        if options
            .get("enable_thinking")
            .is_some_and(|value| value != false)
        {
            return Err("only non-thinking generation is supported".to_owned());
        }
        options.insert("enable_thinking".to_owned(), Value::Bool(false));
        template.render(&messages.to_string(), "null", &request.to_string())
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
    fn refuses_tool_call_messages() {
        assert!(
            render(
                r#"[{"role":"assistant","content":"","tool_calls":[]}]"#,
                "{}"
            )
            .is_err()
        );
    }
    #[test]
    fn refuses_tool_role_messages() {
        assert!(render(r#"[{"role":"tool","content":"result"}]"#, "{}").is_err());
    }
    #[test]
    fn renders_non_thinking_and_refuses_unsupported_requests() {
        assert_eq!(
            render(
                r#"[{"role":"user","content":"Hello"}]"#,
                r#"{"add_generation_prompt":true}"#
            )
            .unwrap(),
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        assert!(render(r#"[{"role":"user","content":[]}]"#, "{}").is_err());
        assert!(render("[]", r#"{"enable_thinking":true}"#).is_err());
        assert!(QwenChat::options(r#"{"unknown":true}"#).is_err());
        assert_eq!(
            TextParser.parse("hello").unwrap(),
            [Event::TextDelta("hello".to_owned())]
        );
    }
}
