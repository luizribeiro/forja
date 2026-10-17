#![allow(clippy::same_length_and_capacity)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "chat-format-component",
});

use exports::l9o::gpu::chat_format::{ChatInfo, Event, GrammarOut, Guest, SamplingDefaults};

struct Component;

impl Guest for Component {
    fn info() -> ChatInfo {
        ChatInfo {
            stop_token_ids: vec![2],
            sampling: SamplingDefaults {
                temperature: Some(0.7),
                top_p: Some(0.8),
                top_k: Some(20),
                repetition_penalty: Some(1.0),
                max_new_tokens: Some(128),
            },
        }
    }

    fn render(
        messages_json: String,
        _tools_json: String,
        options_json: String,
    ) -> Result<String, String> {
        if options_json.contains(r#""loop":true"#) {
            loop {
                std::hint::black_box(());
            }
        }
        if options_json.contains(r#""large":true"#) {
            return Ok("x".repeat(4097));
        }
        if options_json.contains(r#""clock":true"#) {
            return Ok(options_json);
        }
        Ok(format!("rendered:{messages_json}"))
    }

    fn grammar(
        tools_json: String,
        _tool_choice_json: String,
        _response_format_json: String,
    ) -> Result<GrammarOut, String> {
        Ok(GrammarOut {
            grammar: format!("grammar:{tools_json}"),
            trigger: "<tool_call>\n".to_owned(),
        })
    }

    fn parse(_stream: u64, chunk: String) -> Result<Vec<Event>, String> {
        Ok(vec![Event::TextDelta(chunk)])
    }

    fn reset(_stream: u64) {}
}

export!(Component);
