//! Model-agnostic serving policy and protocol support.

#![forbid(unsafe_code)]

mod chat_format;
mod engine;
#[doc(hidden)]
pub mod guidance;
mod http;
mod metrics;
mod prefix;
mod request;
mod response;
mod scheduler;
mod stop;
mod tool_output;

pub use chat_format::{
    ChatEvent, ChatFormat, ChatFormatLimits, ChatInfo, GrammarOutput, SamplingDefaults,
};
pub use engine::{
    Completion, FinishReason, GenerationControl, GenerationError, GenerationEvent, PrefilledPrompt,
    ServingEngine, Usage,
};
pub use http::{ChatService, DrainHandle, chat_router, operational_router, serving_router};
pub use metrics::ServingMetrics;
pub use prefix::PrefixIdentity;
pub use request::{
    ChatCompletionRequest, CompletionParams, CompletionRequest, FunctionCall, FunctionDefinition,
    Message, NamedFunction, NamedToolChoice, ResolvedParams, ResponseFormat, Stop, StreamOptions,
    Tool, ToolCall, ToolChoice, ValidationError, serving_context_ceiling, validate_chat,
    validate_completion,
};
pub use response::{chat_response, text_response};
pub use stop::{StopMatcher, StopOutput};
pub use tool_output::GeneratedToolCall;
