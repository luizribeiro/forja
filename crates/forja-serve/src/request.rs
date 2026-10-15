use std::{collections::BTreeMap, error::Error, fmt};

use forja_config::HostConfig;
use serde::Deserialize;
use serde_json::Value;

use crate::{ChatInfo, SamplingDefaults};

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

/// Validated generation policy.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedParams {
    /// Effective output-token limit.
    pub max_tokens: u32,
    /// Effective sampling temperature.
    pub temperature: f32,
    /// Effective nucleus probability.
    pub top_p: f32,
    /// Effective top-k limit.
    pub top_k: u32,
    /// Optional deterministic seed.
    pub seed: Option<u64>,
    /// Normalized stop strings.
    pub stops: Vec<String>,
    /// Whether the response is streamed.
    pub stream: bool,
}

/// A client-visible invalid-request error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError(String);

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ValidationError {}

/// Validates and resolves a chat-completion request.
///
/// # Errors
///
/// Returns an error when a field is invalid, unsupported, or exceeds a configured ceiling.
pub fn validate_chat(
    request: &ChatCompletionRequest,
    config: &HostConfig,
    defaults: &ChatInfo,
    prompt_tokens: u32,
    max_context: u32,
    vocab: u32,
) -> Result<ResolvedParams, ValidationError> {
    if request.messages.len() > ceiling_usize(config.request.max_messages) {
        return invalid("messages exceed the configured ceiling");
    }
    validate_text(
        checked_text_bytes(request.messages.iter().map(|message| message.content.len()))?,
        config,
    )?;
    resolve(
        &request.params,
        config,
        &defaults.sampling,
        prompt_tokens,
        configured_context(config, max_context),
        vocab,
        true,
    )
}

/// Validates and resolves a text-completion request.
///
/// # Errors
///
/// Returns an error when a field is invalid, unsupported, or exceeds a configured ceiling.
pub fn validate_completion(
    request: &CompletionRequest,
    config: &HostConfig,
    defaults: &ChatInfo,
    prompt_tokens: u32,
    max_context: u32,
    vocab: u32,
) -> Result<ResolvedParams, ValidationError> {
    validate_text(checked_text_bytes([request.prompt.len()])?, config)?;
    resolve(
        &request.params,
        config,
        &defaults.sampling,
        prompt_tokens,
        configured_context(config, max_context),
        vocab,
        false,
    )
}

/// Resolves request values against model and host defaults.
///
/// # Errors
///
/// Returns an error when a value is internally inconsistent or outside its valid range.
fn resolve(
    params: &CompletionParams,
    config: &HostConfig,
    defaults: &SamplingDefaults,
    prompt_tokens: u32,
    max_context: u32,
    vocab: u32,
    chat: bool,
) -> Result<ResolvedParams, ValidationError> {
    reject_unsupported(params)?;
    if params.max_tokens.is_some() && params.max_completion_tokens.is_some() {
        return invalid("max_tokens and max_completion_tokens conflict");
    }
    if !chat && params.max_completion_tokens.is_some() {
        return invalid("max_completion_tokens is only supported for chat completions");
    }
    if prompt_tokens >= max_context {
        return invalid("prompt exceeds the configured context ceiling");
    }
    let requested = params
        .max_completion_tokens
        .or(params.max_tokens)
        .or(defaults.max_new_tokens)
        .unwrap_or(config.request.max_output_tokens);
    if requested == 0 {
        return invalid("output token limit must be positive");
    }
    let temperature = params.temperature.or(defaults.temperature).unwrap_or(1.0);
    let top_p = params.top_p.or(defaults.top_p).unwrap_or(1.0);
    if !temperature.is_finite() || temperature < 0.0 {
        return invalid("temperature must be finite and nonnegative");
    }
    if !(top_p.is_finite() && 0.0 < top_p && top_p <= 1.0) {
        return invalid("top_p must be finite and in (0, 1]");
    }
    let remaining_context = max_context.checked_sub(prompt_tokens).ok_or_else(|| {
        ValidationError("prompt exceeds the configured context ceiling".to_owned())
    })?;
    let stops = normalize_stops(params.stop.as_ref());
    if stops.len() > ceiling_usize(config.request.max_stops) || stops.iter().any(String::is_empty) {
        return invalid("stop strings exceed the configured ceiling or are empty");
    }
    Ok(ResolvedParams {
        max_tokens: requested
            .min(config.request.max_output_tokens)
            .min(remaining_context),
        temperature,
        top_p,
        top_k: params.top_k.or(defaults.top_k).unwrap_or(0).min(vocab),
        seed: params.seed,
        stops,
        stream: params.stream,
    })
}

fn configured_context(config: &HostConfig, engine_profile_context: u32) -> u32 {
    config
        .request
        .max_prompt_tokens
        .unwrap_or(engine_profile_context)
        .min(engine_profile_context)
}

/// Resolves the context supported jointly by an engine, profile, and host configuration.
///
/// # Errors
///
/// Returns an error when the profile claims a context larger than the engine supports.
pub fn serving_context_ceiling(
    engine_max_context: u32,
    profile_max_context: u32,
    configured: Option<u32>,
) -> Result<u32, ValidationError> {
    if profile_max_context > engine_max_context {
        return invalid("profile max-context exceeds the engine maximum");
    }
    Ok(configured
        .unwrap_or(engine_max_context)
        .min(profile_max_context))
}

fn checked_text_bytes<T>(lengths: impl IntoIterator<Item = T>) -> Result<u64, ValidationError>
where
    T: TryInto<u64>,
{
    lengths.into_iter().try_fold(0_u64, |sum, length| {
        let length = length
            .try_into()
            .map_err(|_| ValidationError("request text size overflowed".to_owned()))?;
        sum.checked_add(length)
            .ok_or_else(|| ValidationError("request text size overflowed".to_owned()))
    })
}

fn validate_text(bytes: u64, config: &HostConfig) -> Result<(), ValidationError> {
    if bytes > config.request.max_text_bytes.get() {
        invalid("request text exceeds the configured ceiling")
    } else {
        Ok(())
    }
}

fn ceiling_usize(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

fn reject_unsupported(params: &CompletionParams) -> Result<(), ValidationError> {
    let unsupported = params.n.is_some_and(|value| value != 1)
        || params.presence_penalty.is_some_and(|value| value != 0.0)
        || params.frequency_penalty.is_some_and(|value| value != 0.0)
        || params
            .repetition_penalty
            .is_some_and(|value| value.to_bits() != 1.0_f32.to_bits())
        || params.logprobs == Some(true)
        || params.top_logprobs.is_some_and(|value| value != 0)
        || params
            .logit_bias
            .as_ref()
            .is_some_and(|value| !value.is_empty())
        || params.best_of.is_some_and(|value| value != 1)
        || params.echo == Some(true)
        || params.suffix.is_some()
        || params.tools.as_ref().is_some_and(|value| !value.is_empty())
        || params.tool_choice.is_some()
        || params.response_format.is_some();
    if unsupported {
        invalid("request contains an unsupported non-default field")
    } else {
        Ok(())
    }
}

fn normalize_stops(stop: Option<&Stop>) -> Vec<String> {
    match stop {
        None => Vec::new(),
        Some(Stop::One(stop)) => vec![stop.clone()],
        Some(Stop::Many(stops)) => stops.clone(),
    }
}

fn invalid<T>(message: &str) -> Result<T, ValidationError> {
    Err(ValidationError(message.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(params: CompletionParams) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "qwen".to_owned(),
            messages: vec![Message {
                role: "user".to_owned(),
                content: "hello".to_owned(),
            }],
            params,
        }
    }

    fn resolve_chat(
        params: &CompletionParams,
        defaults: &ChatInfo,
        prompt_tokens: u32,
        max_context: u32,
    ) -> Result<ResolvedParams, ValidationError> {
        resolve(
            params,
            &HostConfig::default(),
            &defaults.sampling,
            prompt_tokens,
            max_context,
            151_936,
            true,
        )
    }

    fn info(sampling: SamplingDefaults) -> ChatInfo {
        ChatInfo {
            stop_token_ids: Vec::new(),
            sampling,
        }
    }

    fn default_info() -> ChatInfo {
        info(SamplingDefaults::default())
    }

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

    #[test]
    fn request_values_override_defaults_and_context_clamps_output() {
        let defaults = info(SamplingDefaults {
            temperature: Some(0.7),
            top_p: Some(0.8),
            top_k: Some(20),
            repetition_penalty: Some(1.05),
            max_new_tokens: None,
        });
        let params = CompletionParams {
            max_completion_tokens: Some(100),
            temperature: Some(0.2),
            top_k: Some(7),
            ..CompletionParams::default()
        };
        let resolved = resolve_chat(&params, &defaults, 32_700, 32_768).unwrap();
        assert_eq!(resolved.max_tokens, 68);
        assert_eq!(resolved.temperature.to_bits(), 0.2_f32.to_bits());
        assert_eq!(resolved.top_p.to_bits(), 0.8_f32.to_bits());
        assert_eq!(resolved.top_k, 7);
    }

    #[test]
    fn conflicting_output_token_fields_are_rejected() {
        let params = CompletionParams {
            max_tokens: Some(1),
            max_completion_tokens: Some(1),
            ..CompletionParams::default()
        };
        let error = resolve_chat(&params, &default_info(), 1, 32_768).unwrap_err();
        assert_eq!(
            error.to_string(),
            "max_tokens and max_completion_tokens conflict"
        );
    }

    #[test]
    fn component_output_limit_is_used_as_the_request_default() {
        let defaults = info(SamplingDefaults {
            max_new_tokens: Some(7),
            ..SamplingDefaults::default()
        });
        let resolved = resolve_chat(&CompletionParams::default(), &defaults, 1, 32_768).unwrap();
        assert_eq!(resolved.max_tokens, 7);
    }

    #[test]
    fn nondefault_completion_count_is_rejected() {
        let params = CompletionParams {
            n: Some(2),
            ..CompletionParams::default()
        };
        let error = resolve_chat(&params, &default_info(), 1, 32_768).unwrap_err();
        assert_eq!(
            error.to_string(),
            "request contains an unsupported non-default field"
        );
    }

    #[test]
    fn nondefault_repetition_penalty_is_rejected() {
        let params = CompletionParams {
            repetition_penalty: Some(1.05),
            ..CompletionParams::default()
        };
        let error = resolve_chat(&params, &default_info(), 1, 32_768).unwrap_err();
        assert_eq!(
            error.to_string(),
            "request contains an unsupported non-default field"
        );
    }

    #[test]
    fn validates_text_completions() {
        let request = CompletionRequest {
            model: "qwen".to_owned(),
            prompt: "hello".to_owned(),
            params: CompletionParams {
                max_tokens: Some(7),
                ..CompletionParams::default()
            },
        };
        let resolved = validate_completion(
            &request,
            &HostConfig::default(),
            &default_info(),
            1,
            32_768,
            151_936,
        )
        .unwrap();
        assert_eq!(resolved.max_tokens, 7);
    }

    #[test]
    fn aggregate_text_size_overflow_is_rejected() {
        let error = checked_text_bytes([u64::MAX, 1]).unwrap_err();
        assert_eq!(error.to_string(), "request text size overflowed");
    }

    #[test]
    fn message_count_ceiling_is_rejected() {
        let mut config = HostConfig::default();
        config.request.max_messages = 0;
        let error = validate_chat(
            &chat(CompletionParams::default()),
            &config,
            &default_info(),
            1,
            32_768,
            151_936,
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "messages exceed the configured ceiling");
    }

    #[test]
    fn aggregate_text_size_ceiling_is_rejected() {
        let config = HostConfig::default();
        let limit = usize::try_from(config.request.max_text_bytes.get()).unwrap();
        let request = ChatCompletionRequest {
            model: "qwen".to_owned(),
            messages: vec![
                Message {
                    role: "user".to_owned(),
                    content: "a".repeat(limit),
                },
                Message {
                    role: "user".to_owned(),
                    content: "b".to_owned(),
                },
            ],
            params: CompletionParams::default(),
        };
        let error =
            validate_chat(&request, &config, &default_info(), 1, 32_768, 151_936).unwrap_err();
        assert_eq!(
            error.to_string(),
            "request text exceeds the configured ceiling"
        );
    }

    #[test]
    fn context_ceiling_is_rejected() {
        let error = validate_chat(
            &chat(CompletionParams::default()),
            &HostConfig::default(),
            &default_info(),
            32_768,
            32_768,
            151_936,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "prompt exceeds the configured context ceiling"
        );
    }

    #[test]
    fn stop_count_ceiling_is_rejected() {
        let params = CompletionParams {
            stop: Some(Stop::Many(vec!["stop".to_owned(); 5])),
            ..CompletionParams::default()
        };
        let error = validate_chat(
            &chat(params),
            &HostConfig::default(),
            &default_info(),
            1,
            32_768,
            151_936,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "stop strings exceed the configured ceiling or are empty"
        );
    }

    #[test]
    fn empty_stop_is_rejected() {
        let params = CompletionParams {
            stop: Some(Stop::One(String::new())),
            ..CompletionParams::default()
        };
        let error = validate_chat(
            &chat(params),
            &HostConfig::default(),
            &default_info(),
            1,
            32_768,
            151_936,
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            "stop strings exceed the configured ceiling or are empty"
        );
    }

    #[test]
    fn context_ceiling_uses_every_loaded_boundary() {
        assert_eq!(
            serving_context_ceiling(32_768, 16_384, None).unwrap(),
            16_384
        );
        assert_eq!(
            serving_context_ceiling(32_768, 16_384, Some(8_192)).unwrap(),
            8_192
        );
        assert!(serving_context_ceiling(4_096, 8_192, None).is_err());
    }

    #[test]
    fn top_k_is_clamped_to_the_engine_vocabulary() {
        let defaults = SamplingDefaults {
            top_k: Some(200),
            ..SamplingDefaults::default()
        };
        let resolved = resolve(
            &CompletionParams::default(),
            &HostConfig::default(),
            &defaults,
            1,
            32_768,
            100,
            true,
        )
        .unwrap();
        assert_eq!(resolved.top_k, 100);
    }
}
