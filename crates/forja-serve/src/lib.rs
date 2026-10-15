//! Model-agnostic serving policy and protocol support.

#![forbid(unsafe_code)]

mod chat_format;
mod request;
mod stop;

use std::{error::Error, fs::File, path::Path};

use serde::Deserialize;

pub use chat_format::{ChatFormat, ChatFormatLimits, ChatInfo, SamplingDefaults};
pub use forja_chat_format::ChatTemplate;
pub use request::{
    ChatCompletionRequest, CompletionParams, CompletionRequest, Message, ResolvedParams, Stop,
    ValidationError, serving_context_ceiling, validate_chat, validate_completion,
};
pub use stop::{StopMatcher, StopOutput};

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
}
