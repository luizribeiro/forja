//! Model-agnostic serving policy and protocol support.

#![forbid(unsafe_code)]

mod request;

use std::{error::Error, fs::File, path::Path};

use minijinja::{AutoEscape, Environment, UndefinedBehavior};
use serde::Deserialize;
use serde_json::{Map, Value};

pub use request::{ChatCompletionRequest, CompletionParams, CompletionRequest, Message, Stop};

/// A compiled publisher-provided chat template.
pub struct ChatTemplate {
    environment: Environment<'static>,
}

impl ChatTemplate {
    /// Compiles a chat template with Hugging Face-compatible helpers.
    ///
    /// # Errors
    ///
    /// Returns a template syntax error.
    pub fn compile(source: String) -> Result<Self, minijinja::Error> {
        let mut environment = Environment::new();
        environment.set_auto_escape_callback(|_| AutoEscape::None);
        environment.set_undefined_behavior(UndefinedBehavior::Strict);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.set_fuel(Some(1_000_000));
        environment.add_template_owned("chat", source)?;
        Ok(Self { environment })
    }

    /// Renders messages, optional tools, and model special-token globals.
    ///
    /// # Errors
    ///
    /// Returns a template lookup or rendering error.
    pub fn render(
        &self,
        messages: &[Value],
        tools: Option<&Value>,
        special_tokens: &Map<String, Value>,
    ) -> Result<String, minijinja::Error> {
        let mut context = special_tokens.clone();
        context.insert("messages".to_owned(), Value::Array(messages.to_vec()));
        context.insert("add_generation_prompt".to_owned(), Value::Bool(true));
        if let Some(tools) = tools {
            context.insert("tools".to_owned(), tools.clone());
        }
        self.environment.get_template("chat")?.render(context)
    }
}

/// One or several token identifiers read from a model generation config.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum TokenIds {
    /// One token identifier.
    One(u32),
    /// Several token identifiers.
    Many(Vec<u32>),
}

/// Supported defaults from a Hugging Face `generation_config.json` file.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default)]
pub struct GenerationConfig {
    /// Padding token identifier.
    pub pad_token_id: Option<u32>,
    /// Whether sampling is enabled.
    pub do_sample: Option<bool>,
    /// Token identifiers that finish a generation.
    pub eos_token_id: Option<TokenIds>,
    /// Repetition penalty requested by the model publisher.
    pub repetition_penalty: Option<f32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability.
    pub top_p: Option<f32>,
    /// Number of highest-scoring tokens retained before sampling.
    pub top_k: Option<u32>,
}

impl GenerationConfig {
    /// Reads model generation defaults from a JSON file.
    ///
    /// # Errors
    ///
    /// Returns an I/O error or a typed JSON decoding error.
    pub fn read(path: &Path) -> Result<Self, Box<dyn Error + Send + Sync>> {
        Ok(serde_json::from_reader(File::open(path)?)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_qwen3_coder_defaults() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen3-coder-generation-config.json");
        let config = GenerationConfig::read(&path).unwrap();
        assert_eq!(config.pad_token_id, Some(151_643));
        assert_eq!(config.do_sample, Some(true));
        assert_eq!(
            config.eos_token_id,
            Some(TokenIds::Many(vec![151_645, 151_643]))
        );
        assert_eq!(config.repetition_penalty, Some(1.05));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.top_p, Some(0.8));
        assert_eq!(config.top_k, Some(20));
    }

    #[test]
    fn enables_python_mapping_methods() {
        let template =
            ChatTemplate::compile("{{ value.keys() | list | join(',') }}".to_owned()).unwrap();
        let messages = Vec::new();
        let mut globals = Map::new();
        globals.insert("value".to_owned(), serde_json::json!({"key": 1}));
        assert_eq!(template.render(&messages, None, &globals).unwrap(), "key");
    }
}
