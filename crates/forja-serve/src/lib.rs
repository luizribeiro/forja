//! Model-agnostic serving policy and protocol support.

#![forbid(unsafe_code)]

mod chat_format;
mod engine;
mod request;
mod stop;

pub use chat_format::{ChatFormat, ChatFormatLimits, ChatInfo, SamplingDefaults};
pub use engine::{GenerationError, PrefilledPrompt, ServingEngine};
pub use request::{
    ChatCompletionRequest, CompletionParams, CompletionRequest, Message, ResolvedParams, Stop,
    ValidationError, serving_context_ceiling, validate_chat, validate_completion,
};
pub use stop::{StopMatcher, StopOutput};
