//! Qwen3-Coder prompt formatting and generation defaults.

#![cfg_attr(not(target_family = "wasm"), forbid(unsafe_code))]

use forja_chat_format::{ChatTemplate, qwen3_coder_xml};

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
    let template = ChatTemplate::compile(TEMPLATE.to_owned()).map_err(|error| error.to_string())?;
    qwen3_coder_xml::render(&template, messages_json, tools_json, options_json)
}

#[cfg(target_family = "wasm")]
#[allow(clippy::same_length_and_capacity)]
mod component {
    use std::{
        collections::BTreeMap,
        sync::{Mutex, OnceLock},
    };

    wit_bindgen::generate!({
        path: "../../wit",
        world: "chat-format-component",
    });

    use exports::l9o::gpu::chat_format::{
        ChatInfo, Event as WitEvent, GrammarOut, Guest, SamplingDefaults,
    };

    static PARSERS: OnceLock<Mutex<BTreeMap<u64, super::qwen3_coder_xml::Parser>>> =
        OnceLock::new();

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

        fn grammar(tools: String, choice: String, response: String) -> Result<GrammarOut, String> {
            super::qwen3_coder_xml::grammar(&tools, &choice, &response)
                .map(|(grammar, trigger)| GrammarOut { grammar, trigger })
        }

        fn parse(stream: u64, chunk: String) -> Result<Vec<WitEvent>, String> {
            let parsers = PARSERS.get_or_init(|| Mutex::new(BTreeMap::new()));
            let mut parsers = parsers
                .lock()
                .map_err(|_| "parser state is unavailable".to_owned())?;
            parsers
                .entry(stream)
                .or_default()
                .parse(&chunk)
                .map(|events| {
                    events
                        .into_iter()
                        .map(|event| match event {
                            super::qwen3_coder_xml::Event::TextDelta(value) => {
                                WitEvent::TextDelta(value)
                            }
                            super::qwen3_coder_xml::Event::CallStart(value) => {
                                WitEvent::CallStart(value)
                            }
                            super::qwen3_coder_xml::Event::ArgumentDelta(name, value) => {
                                WitEvent::ArgumentDelta((name, value))
                            }
                            super::qwen3_coder_xml::Event::CallEnd => WitEvent::CallEnd,
                        })
                        .collect()
                })
        }

        fn reset(stream: u64) {
            if let Some(parsers) = PARSERS.get()
                && let Ok(mut parsers) = parsers.lock()
            {
                parsers.remove(&stream);
            }
        }
    }

    export!(Component);
}
