//! Hugging Face chat-template compatibility checks.

use std::{env, error::Error, fs, path::PathBuf};

use forja_serve::ChatTemplate;
use serde::Deserialize;
use serde_json::{Map, Value};
use tokenizers::Tokenizer;

#[derive(Deserialize)]
struct Golden {
    name: String,
    messages: Vec<Value>,
    rendered: String,
    token_ids: Vec<u32>,
}

#[test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
fn qwen3_coder_matches_hugging_face_chat_goldens() -> Result<(), Box<dyn Error + Send + Sync>> {
    let root = PathBuf::from(env::var_os("FORJA_MODELS").ok_or("FORJA_MODELS is not set")?);
    let model = root.join("Qwen3-Coder-30B-A3B-Instruct-4bit");
    let template = ChatTemplate::compile(fs::read_to_string(model.join("chat_template.jinja"))?)?;
    let tokenizer = Tokenizer::from_file(model.join("tokenizer.json"))?;
    let goldens: Vec<Golden> =
        serde_json::from_str(include_str!("fixtures/qwen3-coder-chat-template.json"))?;
    for golden in goldens {
        let rendered = template.render(&golden.messages, None, &Map::new())?;
        assert_eq!(rendered, golden.rendered, "{} rendering", golden.name);
        let encoded = tokenizer.encode(rendered, false)?;
        assert_eq!(
            encoded.get_ids(),
            golden.token_ids,
            "{} tokens",
            golden.name
        );
    }
    Ok(())
}
