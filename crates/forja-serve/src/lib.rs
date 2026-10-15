//! Model-agnostic serving policy and protocol support.

#![forbid(unsafe_code)]

mod chat_format;
mod engine;
mod http;
mod request;
mod response;
mod stop;

pub use chat_format::{ChatFormat, ChatFormatLimits, ChatInfo, SamplingDefaults};
pub use engine::{
    Completion, FinishReason, GenerationControl, GenerationError, GenerationEvent, PrefilledPrompt,
    ServingEngine, Usage,
};
pub use http::{ChatService, chat_router};
pub use request::{
    ChatCompletionRequest, CompletionParams, CompletionRequest, Message, ResolvedParams, Stop,
    StreamOptions, ValidationError, serving_context_ceiling, validate_chat, validate_completion,
};
pub use response::{chat_response, text_response};
pub use stop::{StopMatcher, StopOutput};
