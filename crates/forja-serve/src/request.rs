use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    fmt,
};

use forja_config::{Choice, HostConfig};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ChatInfo, SamplingDefaults};

const MAX_SCHEMA_VISITS: usize = 65_536;

macro_rules! openai_record {
    ($name:ident { $($field:ident: $kind:ty),+ $(,)? }) => {
        #[allow(missing_docs)]
        #[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
        #[serde(deny_unknown_fields)]
        pub struct $name { $(pub $field: $kind),+ }
    };
}

openai_record!(Message { role: String, content: Option<String>, tool_calls: Option<Vec<ToolCall>>, tool_call_id: Option<String> });
openai_record!(Tool {
    r#type: String,
    function: FunctionDefinition
});
openai_record!(FunctionDefinition { name: String, description: Option<String>, parameters: Value });
openai_record!(ToolCall {
    id: String,
    r#type: String,
    function: FunctionCall
});
openai_record!(FunctionCall {
    name: String,
    arguments: String
});

#[allow(missing_docs)]
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    Named(NamedToolChoice),
}

openai_record!(NamedToolChoice {
    r#type: String,
    function: NamedFunction
});
openai_record!(NamedFunction { name: String });
openai_record!(ResponseFormat { r#type: String });

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
    /// Optional streaming response controls.
    pub stream_options: Option<StreamOptions>,
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
    /// Optional tool definitions.
    pub tools: Option<Vec<Tool>>,
    /// Optional tool selection.
    pub tool_choice: Option<ToolChoice>,
    /// Accepted API compatibility flag; generation remains single-call.
    pub parallel_tool_calls: Option<bool>,
    /// Structured response selection; only text is accepted.
    pub response_format: Option<ResponseFormat>,
}

/// Optional controls for streaming usage emission.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct StreamOptions {
    /// Emit an empty-choices usage chunk before the terminal marker.
    pub include_usage: bool,
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
pub struct ValidationError {
    message: String,
    too_large: bool,
    context_length: bool,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
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
    let params = validate_chat_input(request, config, defaults, vocab)?;
    validate_prompt(
        params,
        prompt_tokens,
        max_context,
        config.request.max_prompt_tokens,
    )
}

pub(crate) fn validate_chat_input(
    request: &ChatCompletionRequest,
    config: &HostConfig,
    defaults: &ChatInfo,
    vocab: u32,
) -> Result<ResolvedParams, ValidationError> {
    if request.messages.len() > ceiling_usize(config.request.max_messages) {
        return invalid("messages exceed the configured ceiling");
    }
    if request.messages.iter().any(|message| {
        serde_json::to_vec(message)
            .ok()
            .and_then(|bytes| u64::try_from(bytes.len()).ok())
            .is_none_or(|bytes| bytes > config.request.max_message_bytes.get())
    }) {
        return too_large("message bytes exceed the configured ceiling");
    }
    validate_text(
        checked_text_bytes(
            request
                .messages
                .iter()
                .filter_map(|message| message.content.as_ref().map(String::len)),
        )?,
        config,
    )?;
    validate_messages(&request.messages)?;
    validate_tools(request, config)?;
    resolve(&request.params, config, &defaults.sampling, vocab, true)
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
    let params = validate_completion_input(request, config, defaults, vocab)?;
    validate_prompt(
        params,
        prompt_tokens,
        max_context,
        config.request.max_prompt_tokens,
    )
}

pub(crate) fn validate_completion_input(
    request: &CompletionRequest,
    config: &HostConfig,
    defaults: &ChatInfo,
    vocab: u32,
) -> Result<ResolvedParams, ValidationError> {
    validate_text(checked_text_bytes([request.prompt.len()])?, config)?;
    resolve(&request.params, config, &defaults.sampling, vocab, false)
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
    vocab: u32,
    chat: bool,
) -> Result<ResolvedParams, ValidationError> {
    reject_unsupported(params, chat)?;
    if params.max_tokens.is_some() && params.max_completion_tokens.is_some() {
        return invalid("max_tokens and max_completion_tokens conflict");
    }
    if !chat && params.max_completion_tokens.is_some() {
        return invalid("max_completion_tokens is only supported for chat completions");
    }
    let requested = params
        .max_completion_tokens
        .or(params.max_tokens)
        .or(defaults.max_new_tokens)
        .unwrap_or(config.request.default_output_tokens);
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
    let stops = normalize_stops(params.stop.as_ref());
    if stops.len() > ceiling_usize(config.request.max_stops)
        || stops.iter().any(|stop| {
            stop.is_empty()
                || u64::try_from(stop.len())
                    .map_or(true, |bytes| bytes > config.request.max_stop_bytes.get())
        })
    {
        return invalid("stop strings exceed the configured ceiling or are empty");
    }
    Ok(ResolvedParams {
        max_tokens: requested.min(config.request.max_output_tokens),
        temperature,
        top_p,
        top_k: params.top_k.or(defaults.top_k).unwrap_or(0).min(vocab),
        seed: params.seed,
        stops,
        stream: params.stream,
    })
}

pub(crate) fn validate_prompt(
    params: ResolvedParams,
    prompt_tokens: u32,
    max_context: u32,
    max_prompt_tokens: Option<u32>,
) -> Result<ResolvedParams, ValidationError> {
    if max_prompt_tokens.is_some_and(|ceiling| prompt_tokens > ceiling) {
        return invalid("prompt exceeds the configured prompt-token ceiling");
    }
    let total = u64::from(prompt_tokens)
        .checked_add(u64::from(params.max_tokens))
        .ok_or_else(|| context_length_error(prompt_tokens, params.max_tokens, max_context))?;
    if total > u64::from(max_context) {
        return Err(context_length_error(
            prompt_tokens,
            params.max_tokens,
            max_context,
        ));
    }
    Ok(params)
}

/// Computes the pages-worth of tokens admission must reserve before streaming starts.
#[cfg_attr(not(test), allow(dead_code, reason = "used by paged admission"))]
pub(crate) fn reservation_tokens(
    prompt_tokens: u32,
    cached_tokens: u32,
    params: &ResolvedParams,
) -> Result<u64, ValidationError> {
    let suffix = prompt_tokens
        .checked_sub(cached_tokens)
        .ok_or_else(|| ValidationError {
            message: "cached tokens exceed the prompt length".to_owned(),
            too_large: false,
            context_length: false,
        })?;
    u64::from(suffix)
        .checked_add(u64::from(params.max_tokens))
        .ok_or_else(|| ValidationError {
            message: "reservation token count overflowed".to_owned(),
            too_large: true,
            context_length: false,
        })
}

/// Resolves the context supported jointly by an engine, profile, and host configuration.
///
/// # Errors
///
/// Returns an error when the profile claims a context larger than the engine supports.
pub fn serving_context_ceiling(
    engine_max_context: u32,
    profile_max_context: u32,
    configured: Choice<std::num::NonZeroU32>,
) -> Result<u32, ValidationError> {
    if profile_max_context > engine_max_context {
        return invalid("profile max-context exceeds the engine maximum");
    }
    let Choice::Fixed(configured) = configured else {
        return Ok(profile_max_context);
    };
    if configured.get() > engine_max_context {
        return Err(ValidationError {
            message: format!(
                "serving.max_context {} exceeds the engine maximum {engine_max_context}",
                configured.get()
            ),
            too_large: false,
            context_length: false,
        });
    }
    if configured.get() > profile_max_context {
        return Err(ValidationError {
            message: format!(
                "serving.max_context {} exceeds the profile maximum {profile_max_context}",
                configured.get()
            ),
            too_large: false,
            context_length: false,
        });
    }
    Ok(configured.get())
}

fn checked_text_bytes<T>(lengths: impl IntoIterator<Item = T>) -> Result<u64, ValidationError>
where
    T: TryInto<u64>,
{
    lengths.into_iter().try_fold(0_u64, |sum, length| {
        let length = length.try_into().map_err(|_| ValidationError {
            message: "request text size overflowed".to_owned(),
            too_large: false,
            context_length: false,
        })?;
        sum.checked_add(length).ok_or_else(|| ValidationError {
            message: "request text size overflowed".to_owned(),
            too_large: false,
            context_length: false,
        })
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

fn reject_unsupported(params: &CompletionParams, chat: bool) -> Result<(), ValidationError> {
    if params.stream_options.is_some() && !params.stream {
        return invalid("stream_options requires stream=true");
    }
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
        || (!chat
            && (params.tools.is_some()
                || params.tool_choice.is_some()
                || params.parallel_tool_calls.is_some()
                || params.response_format.is_some()));
    if unsupported {
        invalid("request contains an unsupported non-default field")
    } else {
        Ok(())
    }
}

fn validate_messages(messages: &[Message]) -> Result<(), ValidationError> {
    let invalid_message = messages.iter().any(|message| match message.role.as_str() {
        "system" | "user" => {
            message.content.is_none()
                || message.tool_calls.is_some()
                || message.tool_call_id.is_some()
        }
        "assistant" => {
            message.tool_call_id.is_some()
                || (message.content.is_none() && message.tool_calls.is_none())
                || message.tool_calls.as_ref().is_some_and(|calls| {
                    calls.len() != 1
                        || calls.iter().any(|call| {
                            call.r#type != "function"
                                || !valid_name(&call.function.name)
                                || serde_json::from_str::<Value>(&call.function.arguments).is_err()
                        })
                })
        }
        "tool" => {
            message.content.is_none()
                || message.tool_calls.is_some()
                || message.tool_call_id.as_ref().is_none_or(String::is_empty)
        }
        _ => true,
    });
    if invalid_message {
        invalid("message role or tool fields are inconsistent")
    } else {
        Ok(())
    }
}

fn validate_tools(
    request: &ChatCompletionRequest,
    config: &HostConfig,
) -> Result<(), ValidationError> {
    let tools = request.params.tools.as_deref().unwrap_or_default();
    let bytes = serde_json::to_vec(tools)
        .ok()
        .and_then(|json| u64::try_from(json.len()).ok());
    if tools.len() > ceiling_usize(config.request.max_tools) {
        return invalid("tool count exceeds the configured ceiling");
    }
    if bytes.is_none_or(|bytes| bytes > config.request.max_tool_bytes.get()) {
        return too_large("tool bytes exceed the configured ceiling");
    }
    let mut names = BTreeSet::new();
    if tools.iter().any(|tool| {
        tool.r#type != "function"
            || !valid_name(&tool.function.name)
            || !names.insert(&tool.function.name)
            || !valid_schema(&tool.function.parameters, config.request.max_schema_depth)
    }) {
        return invalid("tool definition is invalid");
    }
    if request
        .params
        .response_format
        .as_ref()
        .is_some_and(|format| format.r#type != "text")
    {
        return invalid("response_format must be text");
    }
    let valid_choice = match request.params.tool_choice.as_ref() {
        None => true,
        Some(ToolChoice::Mode(mode)) => {
            mode == "none" || (!tools.is_empty() && matches!(mode.as_str(), "auto" | "required"))
        }
        Some(ToolChoice::Named(choice)) => {
            choice.r#type == "function"
                && tools
                    .iter()
                    .any(|tool| tool.function.name == choice.function.name)
        }
    };
    if valid_choice {
        Ok(())
    } else {
        invalid("tool_choice is inconsistent with tools")
    }
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
}

fn valid_schema(root: &Value, max_depth: u32) -> bool {
    SchemaValidator {
        root,
        max_depth,
        active_refs: BTreeSet::new(),
        validated_refs: BTreeSet::new(),
        visits: 0,
    }
    .visit(root, 0)
}

struct SchemaValidator<'a> {
    root: &'a Value,
    max_depth: u32,
    active_refs: BTreeSet<String>,
    validated_refs: BTreeSet<String>,
    visits: usize,
}

impl SchemaValidator<'_> {
    fn visit(&mut self, value: &Value, depth: u32) -> bool {
        self.visits += 1;
        if self.visits > MAX_SCHEMA_VISITS {
            return false;
        }
        let Some(schema) = value.as_object() else {
            return false;
        };
        if depth > self.max_depth
            || schema.keys().any(|key| {
                !matches!(
                    key.as_str(),
                    "type"
                        | "properties"
                        | "required"
                        | "description"
                        | "enum"
                        | "items"
                        | "$defs"
                        | "$ref"
                        | "additionalProperties"
                )
            })
        {
            return false;
        }
        if schema
            .get("type")
            .and_then(Value::as_str)
            .is_some_and(|kind| {
                !matches!(
                    kind,
                    "object" | "array" | "string" | "number" | "integer" | "boolean"
                )
            })
        {
            return false;
        }
        if schema
            .get("additionalProperties")
            .is_some_and(|value| value != &Value::Bool(false))
        {
            return false;
        }
        if !valid_enum(schema) {
            return false;
        }
        if !self.visit_reference(schema, depth) {
            return false;
        }
        let maps_valid = ["properties", "$defs"].into_iter().all(|key| {
            schema
                .get(key)
                .and_then(Value::as_object)
                .is_none_or(|map| {
                    map.iter().all(|(name, child)| {
                        valid_name(name)
                            && self
                                .next_depth(depth)
                                .is_some_and(|depth| self.visit(child, depth))
                    })
                })
        });
        let items_valid = schema.get("items").is_none_or(|child| {
            self.next_depth(depth)
                .is_some_and(|depth| self.visit(child, depth))
        });
        maps_valid && items_valid
    }

    fn visit_reference(&mut self, schema: &serde_json::Map<String, Value>, depth: u32) -> bool {
        let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
            return true;
        };
        let Some(pointer) = reference.strip_prefix('#') else {
            return false;
        };
        if !reference.starts_with("#/$defs/") {
            return false;
        }
        if self.validated_refs.contains(reference) {
            return true;
        }
        if !self.active_refs.insert(reference.to_owned()) {
            return false;
        }
        let valid = self.next_depth(depth).is_some_and(|depth| {
            self.root
                .pointer(pointer)
                .is_some_and(|target| self.visit(target, depth))
        });
        self.active_refs.remove(reference);
        if valid {
            self.validated_refs.insert(reference.to_owned());
        }
        valid
    }

    fn next_depth(&self, depth: u32) -> Option<u32> {
        depth
            .checked_add(1)
            .filter(|depth| *depth <= self.max_depth)
    }
}

fn valid_enum(schema: &serde_json::Map<String, Value>) -> bool {
    let Some(values) = schema.get("enum") else {
        return true;
    };
    let Some(values) = values.as_array().filter(|values| !values.is_empty()) else {
        return false;
    };
    match schema.get("type").and_then(Value::as_str) {
        Some("string") => values.iter().all(Value::is_string),
        Some("number") => values.iter().all(Value::is_number),
        Some("integer") => values
            .iter()
            .all(|value| value.as_i64().is_some() || value.as_u64().is_some()),
        Some("boolean") => values.iter().all(Value::is_boolean),
        _ => false,
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
    Err(ValidationError {
        message: message.to_owned(),
        too_large: false,
        context_length: false,
    })
}

fn too_large<T>(message: &str) -> Result<T, ValidationError> {
    Err(ValidationError {
        message: message.to_owned(),
        too_large: true,
        context_length: false,
    })
}

fn context_length_error(prompt: u32, output: u32, maximum: u32) -> ValidationError {
    ValidationError {
        message: format!(
            "maximum context length is {maximum} tokens, but the request has {prompt} prompt tokens and requests {output} completion tokens"
        ),
        too_large: false,
        context_length: true,
    }
}

impl ValidationError {
    pub(crate) const fn is_too_large(&self) -> bool {
        self.too_large
    }

    pub(crate) const fn is_context_length(&self) -> bool {
        self.context_length
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chat(params: CompletionParams) -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "qwen".to_owned(),
            messages: vec![Message {
                role: "user".to_owned(),
                content: Some("hello".to_owned()),
                tool_calls: None,
                tool_call_id: None,
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
        let resolved = resolve(
            params,
            &HostConfig::default(),
            &defaults.sampling,
            151_936,
            true,
        )?;
        validate_prompt(resolved, prompt_tokens, max_context, None)
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
        let tools = r#"{"model":"qwen","messages":[{"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]},{"role":"tool","content":"ok","tool_call_id":"call_0"}],"tools":[{"type":"function","function":{"name":"read_file","description":"read","parameters":{"type":"object"}}}],"tool_choice":"auto","parallel_tool_calls":true,"response_format":{"type":"text"}}"#;
        let mut request: ChatCompletionRequest = serde_json::from_str(tools).unwrap();
        assert_eq!(
            request.params.tools.as_ref().unwrap()[0].function.name,
            "read_file"
        );
        validate_chat_input(&request, &HostConfig::default(), &default_info(), 151_936).unwrap();
        let mut byte_limited = HostConfig::default();
        byte_limited.request.max_tool_bytes = forja_config::ByteSize::new(1);
        assert!(
            validate_chat_input(&request, &byte_limited, &default_info(), 151_936)
                .unwrap_err()
                .is_too_large()
        );
        let duplicate = request.params.tools.as_ref().unwrap()[0].clone();
        request.params.tools.as_mut().unwrap().push(duplicate);
        assert!(
            validate_chat_input(&request, &HostConfig::default(), &default_info(), 151_936)
                .is_err()
        );
        request.params.tools.as_mut().unwrap().pop();
        request.messages[1].tool_call_id = None;
        assert!(
            validate_chat_input(&request, &HostConfig::default(), &default_info(), 151_936)
                .is_err()
        );
        assert!(
            serde_json::from_str::<ChatCompletionRequest>(&tools.replace(
                "\"name\":\"read_file\"",
                "\"name\":\"read_file\",\"unknown\":true"
            ))
            .is_err()
        );
    }

    #[test]
    fn schema_validation_bounds_depth_and_references() {
        let valid = serde_json::json!({"type":"object","properties":{"path":{"$ref":"#/$defs/path"}},"$defs":{"path":{"type":"string"}}});
        assert!(valid_schema(&valid, 2));
        for schema in [
            serde_json::json!({"$ref":"https://example.com/schema"}),
            serde_json::json!({"$defs":{"x":{"$ref":"#/$defs/x"}},"$ref":"#/$defs/x"}),
            serde_json::json!({"type":"string","pattern":".*"}),
        ] {
            assert!(!valid_schema(&schema, 2));
        }
        assert!(!valid_schema(&valid, 0));
    }

    #[test]
    fn schema_validation_memoizes_diamond_references() {
        let mut definitions = serde_json::Map::new();
        definitions.insert("leaf".to_owned(), serde_json::json!({"type":"string"}));
        for level in (0..30).rev() {
            let next = if level == 29 {
                "leaf".to_owned()
            } else {
                format!("level_{}", level + 1)
            };
            definitions.insert(
                format!("level_{level}"),
                serde_json::json!({
                    "type":"object",
                    "properties": {
                        "left":{"$ref":format!("#/$defs/{next}")},
                        "right":{"$ref":format!("#/$defs/{next}")}
                    }
                }),
            );
        }
        let schema = serde_json::json!({
            "$defs": definitions,
            "$ref":"#/$defs/level_0"
        });
        let started = std::time::Instant::now();
        assert!(valid_schema(&schema, 64));
        assert!(started.elapsed() < std::time::Duration::from_millis(100));
    }

    #[test]
    fn schema_validation_refuses_unenforceable_enums() {
        for schema in [
            serde_json::json!({"type":"array","enum":[[]]}),
            serde_json::json!({"type":"object","enum":[{}]}),
            serde_json::json!({"enum":["missing type"]}),
            serde_json::json!({"type":"integer","enum":[1.5]}),
            serde_json::json!({"type":"boolean","enum":[]}),
        ] {
            assert!(!valid_schema(&schema, 64));
        }
        for schema in [
            serde_json::json!({"type":"string","enum":["red","blue"]}),
            serde_json::json!({"type":"number","enum":[1,2.5]}),
            serde_json::json!({"type":"integer","enum":[-1,2]}),
            serde_json::json!({"type":"boolean","enum":[true,false]}),
        ] {
            assert!(valid_schema(&schema, 64));
        }
    }

    #[test]
    fn request_values_override_defaults() {
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
        let resolved = resolve_chat(&params, &defaults, 32_000, 32_768).unwrap();
        assert_eq!(resolved.max_tokens, 100);
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
    fn output_limit_precedence_also_controls_paged_reservation() {
        let mut config = HostConfig::default();
        config.request.default_output_tokens = 17;
        let defaults = info(SamplingDefaults {
            max_new_tokens: Some(33),
            ..SamplingDefaults::default()
        });
        let requested = CompletionParams {
            max_tokens: Some(7),
            ..CompletionParams::default()
        };
        let omitted = CompletionParams::default();
        let no_model_default = default_info();
        for (params, model, expected) in [
            (&requested, &defaults, 7),
            (&omitted, &defaults, 33),
            (&omitted, &no_model_default, 17),
        ] {
            let resolved = resolve(params, &config, &model.sampling, 151_936, true).unwrap();
            assert_eq!(resolved.max_tokens, expected);
            assert_eq!(
                reservation_tokens(2_048, 1_024, &resolved).unwrap(),
                1_024 + u64::from(expected)
            );
            assert!(reservation_tokens(2_048, 2_049, &resolved).is_err());
        }
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
        let mut config = HostConfig::default();
        config.request.max_message_bytes = forja_config::ByteSize::new(2 * 1024 * 1024);
        let limit = usize::try_from(config.request.max_text_bytes.get()).unwrap();
        let request = ChatCompletionRequest {
            model: "qwen".to_owned(),
            messages: vec![
                Message {
                    role: "user".to_owned(),
                    content: Some("a".repeat(limit)),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: "user".to_owned(),
                    content: Some("b".to_owned()),
                    tool_calls: None,
                    tool_call_id: None,
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
    fn individual_message_and_stop_byte_ceilings_are_rejected() {
        let mut config = HostConfig::default();
        config.request.max_message_bytes = forja_config::ByteSize::new(32);
        let message = validate_chat(
            &chat(CompletionParams::default()),
            &config,
            &default_info(),
            1,
            32_768,
            151_936,
        )
        .unwrap_err();
        assert!(message.is_too_large());

        config.request.max_message_bytes = forja_config::ByteSize::new(1024);
        config.request.max_stop_bytes = forja_config::ByteSize::new(3);
        let params = CompletionParams {
            stop: Some(Stop::One("stop".to_owned())),
            ..CompletionParams::default()
        };
        assert!(
            validate_chat(&chat(params), &config, &default_info(), 1, 32_768, 151_936).is_err()
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
        assert!(error.is_context_length());
        assert!(error.to_string().contains("32768 prompt tokens"));

        let lowered = validate_chat(
            &chat(CompletionParams::default()),
            &HostConfig::default(),
            &default_info(),
            4_000,
            4_096,
            151_936,
        )
        .unwrap_err();
        assert!(lowered.is_context_length());
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
            serving_context_ceiling(32_768, 16_384, Choice::Auto).unwrap(),
            16_384
        );
        assert_eq!(
            serving_context_ceiling(
                32_768,
                16_384,
                Choice::Fixed(std::num::NonZeroU32::new(8_192).unwrap()),
            )
            .unwrap(),
            8_192
        );
        assert!(serving_context_ceiling(4_096, 8_192, Choice::Auto).is_err());
        let error = serving_context_ceiling(
            32_768,
            16_384,
            Choice::Fixed(std::num::NonZeroU32::new(20_000).unwrap()),
        )
        .unwrap_err();
        assert!(error.to_string().contains("profile maximum 16384"));
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
            100,
            true,
        )
        .unwrap();
        assert_eq!(resolved.top_k, 100);
    }
}
