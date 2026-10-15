#![allow(clippy::same_length_and_capacity)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "chat-format-component",
});

use exports::l9o::gpu::chat_format::{ChatInfo, Guest, SamplingDefaults};

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
        if options_json == "loop" {
            loop {
                std::hint::black_box(());
            }
        }
        if options_json == "large" {
            return Ok("x".repeat(4097));
        }
        Ok(format!("rendered:{messages_json}"))
    }
}

export!(Component);
