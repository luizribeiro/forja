#![allow(clippy::same_length_and_capacity)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "chat-format-component",
});

use std::{cell::RefCell, collections::BTreeSet};

use exports::l9o::gpu::chat_format::{ChatInfo, Event, GrammarOut, Guest, SamplingDefaults};

thread_local! {
    static PARSERS: RefCell<BTreeSet<u64>> = const { RefCell::new(BTreeSet::new()) };
}

struct Component;

impl Guest for Component {
    fn info() -> ChatInfo {
        ChatInfo {
            stop_token_ids: Vec::new(),
            sampling: SamplingDefaults {
                temperature: Some(0.0),
                top_p: Some(1.0),
                top_k: Some(1),
                repetition_penalty: None,
                max_new_tokens: Some(16),
            },
        }
    }

    fn render(
        _messages_json: String,
        _tools_json: String,
        _options_json: String,
    ) -> Result<String, String> {
        Ok("a".to_owned())
    }

    fn grammar(
        _tools_json: String,
        _tool_choice_json: String,
        _response_format_json: String,
    ) -> Result<GrammarOut, String> {
        Ok(GrammarOut {
            grammar: format!("start: {:?}", "bcda".repeat(64)),
            trigger: "a".to_owned(),
        })
    }

    fn parse(stream: u64, _chunk: String) -> Result<Vec<Event>, String> {
        PARSERS.with(|parsers| {
            if parsers.borrow_mut().insert(stream) {
                Ok(vec![Event::CallStart("tool".to_owned())])
            } else {
                Ok(Vec::new())
            }
        })
    }

    fn reset(stream: u64) {
        PARSERS.with(|parsers| {
            parsers.borrow_mut().remove(&stream);
        });
    }
}

export!(Component);
