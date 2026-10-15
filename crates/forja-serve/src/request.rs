use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::Value;

/// One chat message with string content.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Message {
    /// OpenAI-compatible message role.
    pub role: String,
    /// Message text.
    pub content: String,
}

/// One stop string or a list of stop strings.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum Stop {
    /// One stop string.
    One(String),
    /// Several stop strings.
    Many(Vec<String>),
}

/// Parameters shared by completion request forms.
#[derive(Clone, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct CompletionParams {
    /// Legacy output-token limit.
    pub max_tokens: Option<u32>,
    /// Chat output-token limit.
    pub max_completion_tokens: Option<u32>,
    /// Sampling temperature.
    pub temperature: Option<f32>,
    /// Nucleus-sampling probability.
    pub top_p: Option<f32>,
    /// Documented extension limiting sampling to the highest-scoring tokens.
    pub top_k: Option<u32>,
    /// Reproducible sampling seed.
    pub seed: Option<u64>,
    /// Text stop strings.
    pub stop: Option<Stop>,
    /// Whether to stream response events.
    pub stream: bool,
    /// Number of completions.
    pub n: Option<u32>,
    /// Unsupported presence penalty.
    pub presence_penalty: Option<f32>,
    /// Unsupported frequency penalty.
    pub frequency_penalty: Option<f32>,
    /// Unsupported repetition-penalty extension.
    pub repetition_penalty: Option<f32>,
    /// Unsupported log-probability output.
    pub logprobs: Option<bool>,
    /// Unsupported number of top log probabilities.
    pub top_logprobs: Option<u32>,
    /// Unsupported token bias map.
    pub logit_bias: Option<BTreeMap<String, f32>>,
    /// Unsupported completion candidate count.
    pub best_of: Option<u32>,
    /// Unsupported prompt echo.
    pub echo: Option<bool>,
    /// Unsupported prompt suffix.
    pub suffix: Option<String>,
    /// Unsupported tool definitions.
    pub tools: Option<Vec<Value>>,
    /// Unsupported tool selection.
    pub tool_choice: Option<Value>,
    /// Unsupported structured response selection.
    pub response_format: Option<Value>,
}

/// An OpenAI-compatible chat-completion request.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ChatCompletionRequest {
    /// Requested served model name.
    pub model: String,
    /// Conversation messages.
    pub messages: Vec<Message>,
    /// Generation parameters.
    #[serde(flatten)]
    pub params: CompletionParams,
}

/// An OpenAI-compatible text-completion request.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    /// Requested served model name.
    pub model: String,
    /// Text prompt.
    pub prompt: String,
    /// Generation parameters.
    #[serde(flatten)]
    pub params: CompletionParams,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_json_is_strict_and_accepts_top_k() {
        let request: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":"qwen","messages":[],"top_k":7,"n":1}"#).unwrap();
        assert_eq!(request.params.top_k, Some(7));
        assert!(
            serde_json::from_str::<ChatCompletionRequest>(
                r#"{"model":"qwen","messages":[],"unknown":true}"#
            )
            .is_err()
        );
    }
}
