//! Qwen3-Coder prompt formatting and generation defaults.

#![cfg_attr(not(target_family = "wasm"), forbid(unsafe_code))]

use forja_chat_format::ChatTemplate;

/// Publisher-declared token ids that stop generation.
pub const STOP_TOKEN_IDS: &[u32] = &[151_645, 151_643];
/// Publisher-declared sampling temperature.
pub const TEMPERATURE: f32 = 0.7;
/// Publisher-declared nucleus probability.
pub const TOP_P: f32 = 0.8;
/// Publisher-declared highest-scoring token count.
pub const TOP_K: u32 = 20;
/// Publisher-declared repetition penalty.
pub const REPETITION_PENALTY: f32 = 1.05;

const TEMPLATE: &str = include_str!("chat_template.jinja");

/// Renders a Qwen3-Coder prompt from the chat-format JSON boundary.
///
/// # Errors
///
/// Returns template compilation, JSON, or rendering errors as guest-safe text.
pub fn render(messages_json: &str, tools_json: &str, options_json: &str) -> Result<String, String> {
    ChatTemplate::compile(TEMPLATE.to_owned())
        .map_err(|error| error.to_string())?
        .render(messages_json, tools_json, options_json)
}

#[cfg(target_family = "wasm")]
#[allow(clippy::same_length_and_capacity)]
mod component {
    wit_bindgen::generate!({
        path: "../../wit",
        world: "chat-format-component",
    });

    use exports::l9o::gpu::chat_format::{ChatInfo, Event, GrammarOut, Guest, SamplingDefaults};

    struct Component;

    impl Guest for Component {
        fn info() -> ChatInfo {
            ChatInfo {
                stop_token_ids: super::STOP_TOKEN_IDS.to_vec(),
                sampling: SamplingDefaults {
                    temperature: Some(super::TEMPERATURE),
                    top_p: Some(super::TOP_P),
                    top_k: Some(super::TOP_K),
                    repetition_penalty: Some(super::REPETITION_PENALTY),
                    max_new_tokens: None,
                },
            }
        }

        fn render(
            messages_json: String,
            tools_json: String,
            options_json: String,
        ) -> Result<String, String> {
            super::render(&messages_json, &tools_json, &options_json)
        }

        fn grammar(_: String, _: String, _: String) -> Result<GrammarOut, String> {
            Err("tool grammars are not implemented".to_owned())
        }

        fn parse(_: u64, _: String) -> Result<Vec<Event>, String> {
            Err("tool parsing is not implemented".to_owned())
        }

        fn reset(_: u64) {}
    }

    export!(Component);
}
