use std::{
    collections::VecDeque,
    error::Error,
    fmt,
    fs::File,
    future::{Future, poll_fn},
    io::Read,
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use forja_config::HostConfig;
use forja_core::Backend;
use forja_host::{EngineDecode, EngineDecodeOutput, EngineInfo, EngineRunner, SamplingParams};
use tokenizers::Tokenizer;
use tokio::sync::{Notify, mpsc};

use crate::request::{validate_chat_input, validate_completion_input, validate_prompt};
use crate::{
    ChatCompletionRequest, ChatFormat, CompletionRequest, ResolvedParams, StopMatcher,
    serving_context_ceiling,
};

/// A failure while preparing or running a generation.
#[derive(Debug)]
pub struct GenerationError {
    message: String,
    kind: ErrorKind,
}

impl GenerationError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::Internal,
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::Invalid,
        }
    }

    pub(crate) fn too_large(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::PayloadTooLarge,
        }
    }

    fn unavailable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::Unavailable,
        }
    }

    fn cancelled() -> Self {
        Self {
            message: "generation cancelled".to_owned(),
            kind: ErrorKind::Unavailable,
        }
    }

    pub(crate) const fn kind(&self) -> ErrorKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ErrorKind {
    Invalid,
    PayloadTooLarge,
    Internal,
    Unavailable,
}

impl fmt::Display for GenerationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for GenerationError {}

/// Idempotent cancellation and stream-stall policy for one admitted sequence.
#[derive(Clone)]
pub struct GenerationControl {
    cancellation: Arc<Cancellation>,
    write_stall_timeout: Duration,
}

struct Cancellation {
    cancelled: AtomicBool,
    notify: Notify,
}

impl GenerationControl {
    /// Creates request control with the configured write-stall timeout.
    #[must_use]
    pub fn new(write_stall_timeout: Duration) -> Self {
        Self {
            cancellation: Arc::new(Cancellation {
                cancelled: AtomicBool::new(false),
                notify: Notify::new(),
            }),
            write_stall_timeout,
        }
    }

    /// Requests cancellation. Repeated calls have no additional effect.
    pub fn cancel(&self) {
        if !self.cancellation.cancelled.swap(true, Ordering::AcqRel) {
            self.cancellation.notify.notify_one();
        }
    }

    pub(crate) fn is_cancelled(&self) -> bool {
        self.cancellation.cancelled.load(Ordering::Acquire)
    }
}

/// Device-resident first-token selection and its validated request policy.
pub struct PrefilledPrompt {
    /// First selected token, still resident on the backend.
    pub first: EngineDecodeOutput,
    /// Validated generation policy.
    pub params: ResolvedParams,
    /// Rendered prompt token count.
    pub prompt_tokens: u32,
    /// Engine sampling parameters used by every decode step.
    pub sampling: SamplingParams,
}

trait PromptRenderer: Send {
    fn render_prompt(
        &mut self,
        messages: &str,
        tools: &str,
        options: &str,
    ) -> Result<String, GenerationError>;
}

struct PromptArtifacts<R> {
    tokenizer: Tokenizer,
    renderer: R,
}

impl PromptRenderer for ChatFormat {
    fn render_prompt(
        &mut self,
        messages: &str,
        tools: &str,
        options: &str,
    ) -> Result<String, GenerationError> {
        self.render(messages, tools, options)
            .map_err(|error| GenerationError::new(format!("chat format failed: {error}")))?
            .map_err(GenerationError::invalid)
    }
}

/// Why token generation ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// A model-declared token or request stop string matched.
    Stop,
    /// The validated output-token limit was reached.
    Length,
}

/// Token counts for one completed request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Usage {
    /// Tokens in the rendered prompt.
    pub prompt_tokens: u32,
    /// Sampled tokens, including a token that caused a stop.
    pub completion_tokens: u32,
}

/// Fully decoded output from one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Completion {
    /// Decoded text before any matched stop.
    pub text: String,
    /// Why generation ended.
    pub finish_reason: FinishReason,
    /// Request token counts.
    pub usage: Usage,
}

/// One logical event produced during a streamed generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationEvent {
    /// Initial assistant role for a chat stream.
    Role,
    /// Decoded content safe to expose to the client.
    Content(String),
    /// The terminal choice chunk.
    Finished(FinishReason, Usage),
    /// Optional usage-only chunk.
    Usage(Usage),
    /// Clean stream terminator.
    Done,
    /// One terminal error emitted after response headers.
    Error(String),
}

/// One loaded engine, tokenizer, and model-declared chat format.
pub struct ServingEngine<B: Backend + Send + Sync + 'static> {
    model: String,
    runner: EngineRunner<B>,
    prompt: Arc<StdMutex<PromptArtifacts<ChatFormat>>>,
    chat_info: crate::ChatInfo,
    config: HostConfig,
    info: EngineInfo,
    max_context: u32,
}

impl<B> ServingEngine<B>
where
    B: Backend + Send + Sync + 'static,
{
    /// Joins already-loaded serving artifacts after checking their shared boundaries.
    ///
    /// # Errors
    ///
    /// Returns an error when the profile or tokenizer exceeds the engine contract.
    pub fn new(
        model: String,
        runner: EngineRunner<B>,
        tokenizer: Tokenizer,
        chat_format: ChatFormat,
        config: HostConfig,
        info: EngineInfo,
        profile_max_context: u32,
    ) -> Result<Self, GenerationError> {
        let max_context = serving_context_ceiling(
            info.max_context,
            profile_max_context,
            config.request.max_prompt_tokens,
        )
        .map_err(|error| GenerationError::new(error.to_string()))?;
        if tokenizer
            .get_vocab(true)
            .values()
            .any(|&token| token >= info.vocab)
        {
            return Err(GenerationError::new(
                "tokenizer contains ids outside the engine vocabulary",
            ));
        }
        let chat_info = chat_format.info().clone();
        Ok(Self {
            model,
            runner,
            prompt: Arc::new(StdMutex::new(PromptArtifacts {
                tokenizer,
                renderer: chat_format,
            })),
            chat_info,
            config,
            info,
            max_context,
        })
    }

    /// Returns the only model identifier accepted by this engine.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    pub(crate) fn max_body_bytes(&self) -> usize {
        usize::try_from(self.config.request.max_body_bytes.get()).unwrap_or(usize::MAX)
    }

    pub(crate) const fn queue_length(&self) -> u32 {
        self.config.server.queue_length
    }

    pub(crate) const fn request_deadline(&self) -> Duration {
        self.config.server.request_deadline.get()
    }

    pub(crate) const fn write_stall_timeout(&self) -> Duration {
        self.config.server.sse_write_stall_timeout.get()
    }

    fn tokenizer(&self) -> Result<Tokenizer, GenerationError> {
        self.prompt
            .lock()
            .map(|prompt| prompt.tokenizer.clone())
            .map_err(|_| GenerationError::new("prompt processor lock is poisoned"))
    }

    /// Renders, tokenizes, validates, and prefills one chat request at position zero.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid input, chat-format failure, or engine failure.
    pub async fn prefill_chat(
        &mut self,
        request: &ChatCompletionRequest,
    ) -> Result<PrefilledPrompt, GenerationError> {
        let control = GenerationControl::new(self.write_stall_timeout());
        self.prefill_chat_controlled(request, &control).await
    }

    async fn prefill_chat_controlled(
        &mut self,
        request: &ChatCompletionRequest,
        control: &GenerationControl,
    ) -> Result<PrefilledPrompt, GenerationError> {
        let (tokens, prompt_tokens, params) = prepare_chat_prompt(
            request,
            &self.config,
            &self.chat_info,
            self.max_context,
            self.info.vocab,
            Arc::clone(&self.prompt),
            control,
        )
        .await?;
        self.prefill(tokens, prompt_tokens, params, control).await
    }

    async fn prefill_completion(
        &mut self,
        request: &CompletionRequest,
        control: &GenerationControl,
    ) -> Result<PrefilledPrompt, GenerationError> {
        let (tokens, prompt_tokens, params) = prepare_completion_prompt(
            request,
            &self.config,
            &self.chat_info,
            self.max_context,
            self.info.vocab,
            Arc::clone(&self.prompt),
            control,
        )
        .await?;
        self.prefill(tokens, prompt_tokens, params, control).await
    }

    async fn prefill(
        &mut self,
        tokens: Vec<u32>,
        prompt_tokens: u32,
        params: ResolvedParams,
        control: &GenerationControl,
    ) -> Result<PrefilledPrompt, GenerationError> {
        let sampling = SamplingParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            seed: sampling_seed(params.seed, params.temperature).await?,
        };
        if control.is_cancelled() {
            return Err(GenerationError::cancelled());
        }
        let first = self
            .runner
            .enqueue_decode(EngineDecode {
                tokens: Some(tokens),
                start_pos: 0,
                sampling,
            })
            .await
            .map_err(|error| {
                GenerationError::unavailable(format!("engine prefill failed: {error}"))
            })?
            .map_err(|error| {
                GenerationError::unavailable(format!("engine prefill failed: {error:?}"))
            })?;
        Ok(PrefilledPrompt {
            first,
            params,
            prompt_tokens,
            sampling,
        })
    }

    /// Generates a complete chat response with two decode calls in flight.
    ///
    /// # Errors
    ///
    /// Returns an error when decoding, backend synchronization, or detokenization fails.
    pub async fn complete_chat(
        &mut self,
        request: &ChatCompletionRequest,
    ) -> Result<Completion, GenerationError> {
        let control = GenerationControl::new(self.write_stall_timeout());
        self.complete_chat_controlled(request, &control).await
    }

    pub(crate) async fn complete_chat_controlled(
        &mut self,
        request: &ChatCompletionRequest,
        control: &GenerationControl,
    ) -> Result<Completion, GenerationError> {
        let prefill = self.prefill_chat_controlled(request, control).await?;
        self.decode(prefill, None, control).await
    }

    /// Generates a complete response from an unformatted text prompt.
    ///
    /// # Errors
    ///
    /// Returns an error when validation, tokenization, or engine execution fails.
    pub async fn complete_text(
        &mut self,
        request: &CompletionRequest,
    ) -> Result<Completion, GenerationError> {
        let control = GenerationControl::new(self.write_stall_timeout());
        self.complete_text_controlled(request, &control).await
    }

    pub(crate) async fn complete_text_controlled(
        &mut self,
        request: &CompletionRequest,
        control: &GenerationControl,
    ) -> Result<Completion, GenerationError> {
        let prefill = self.prefill_completion(request, control).await?;
        self.decode(prefill, None, control).await
    }

    /// Streams chat content and terminal events in generation order.
    ///
    /// # Errors
    ///
    /// Returns an error when generation fails or the receiver is abandoned.
    pub async fn stream_chat(
        &mut self,
        request: &ChatCompletionRequest,
        sender: &mpsc::Sender<GenerationEvent>,
    ) -> Result<(), GenerationError> {
        let control = GenerationControl::new(self.write_stall_timeout());
        self.stream_chat_controlled(request, sender, &control).await
    }

    pub(crate) async fn stream_chat_controlled(
        &mut self,
        request: &ChatCompletionRequest,
        sender: &mpsc::Sender<GenerationEvent>,
        control: &GenerationControl,
    ) -> Result<(), GenerationError> {
        let prefill = self.prefill_chat_controlled(request, control).await?;
        let completion = self.decode(prefill, Some(sender), control).await?;
        send_terminal(sender, &request.params, completion, control).await
    }

    /// Streams text-completion content and terminal events in generation order.
    ///
    /// # Errors
    ///
    /// Returns an error when generation fails or the receiver is abandoned.
    pub async fn stream_text(
        &mut self,
        request: &CompletionRequest,
        sender: &mpsc::Sender<GenerationEvent>,
    ) -> Result<(), GenerationError> {
        let control = GenerationControl::new(self.write_stall_timeout());
        self.stream_text_controlled(request, sender, &control).await
    }

    pub(crate) async fn stream_text_controlled(
        &mut self,
        request: &CompletionRequest,
        sender: &mpsc::Sender<GenerationEvent>,
        control: &GenerationControl,
    ) -> Result<(), GenerationError> {
        let prefill = self.prefill_completion(request, control).await?;
        let completion = self.decode(prefill, Some(sender), control).await?;
        send_terminal(sender, &request.params, completion, control).await
    }

    async fn decode(
        &mut self,
        prefill: PrefilledPrompt,
        sender: Option<&mpsc::Sender<GenerationEvent>>,
        control: &GenerationControl,
    ) -> Result<Completion, GenerationError> {
        let mut pending = VecDeque::from([prefill.first]);
        let mut issued = 1_u32;
        let mut consumed = 0_u32;
        let mut position = prefill.prompt_tokens;
        let replay_base = self.runner.metrics().submissions;
        let tokenizer = self.tokenizer()?;
        let mut decoder = tokenizer.decode_stream(true);
        let mut matcher = StopMatcher::new(&prefill.params.stops);
        let mut text = String::new();
        let generation = async {
            while let Some(output) = pending.pop_front() {
                if control.is_cancelled() {
                    return Err(GenerationError::cancelled());
                }
                while pending.len() < 2 && issued < prefill.params.max_tokens {
                    pending.push_back(
                        self.runner
                            .enqueue_decode(EngineDecode {
                                tokens: None,
                                start_pos: position,
                                sampling: prefill.sampling,
                            })
                            .await
                            .map_err(|error| {
                                GenerationError::unavailable(format!(
                                    "engine decode failed: {error}"
                                ))
                            })?
                            .map_err(|error| {
                                GenerationError::unavailable(format!(
                                    "engine decode failed: {error:?}"
                                ))
                            })?,
                    );
                    position = position
                        .checked_add(1)
                        .ok_or_else(|| GenerationError::new("decode position overflowed"))?;
                    issued += 1;
                }
                let bytes = self
                    .runner
                    .read_queued_token(output)
                    .await
                    .map_err(|error| {
                        GenerationError::unavailable(format!(
                            "selected-token read failed: {error:?}"
                        ))
                    })?;
                let token = read_token(&bytes)?;
                consumed += 1;
                if consumed > 1 {
                    let expected = replay_base
                        .checked_add(u64::from(consumed - 1))
                        .ok_or_else(|| GenerationError::new("submission count overflowed"))?;
                    self.runner
                        .wait_for_submissions(expected)
                        .await
                        .map_err(|error| {
                            GenerationError::unavailable(format!(
                                "engine synchronization failed: {error}"
                            ))
                        })?;
                }
                if self.chat_info.stop_token_ids.contains(&token) {
                    return Ok(FinishReason::Stop);
                }
                if let Some(piece) = decoder.step(token).map_err(|error| {
                    GenerationError::new(format!("cannot decode token: {error}"))
                })? {
                    let output = matcher.push(piece.as_bytes()).map_err(|error| {
                        GenerationError::new(format!("decoded invalid UTF-8: {error}"))
                    })?;
                    if emit_output(&mut text, output, sender, control).await? {
                        return Ok(FinishReason::Stop);
                    }
                }
            }
            let output = matcher
                .finish()
                .map_err(|error| GenerationError::new(format!("decoded invalid UTF-8: {error}")))?;
            emit_output(&mut text, output, sender, control).await?;
            Ok(FinishReason::Length)
        }
        .await;
        let discard = finalize_sequence(&mut self.runner, &mut pending).await;
        let finish_reason = match (generation, discard) {
            (Ok(reason), Ok(())) => reason,
            (Err(error), _) | (Ok(_), Err(error)) => return Err(error),
        };
        Ok(Completion {
            text,
            finish_reason,
            usage: Usage {
                prompt_tokens: prefill.prompt_tokens,
                completion_tokens: consumed,
            },
        })
    }
}

async fn emit_output(
    text: &mut String,
    output: crate::StopOutput,
    sender: Option<&mpsc::Sender<GenerationEvent>>,
    control: &GenerationControl,
) -> Result<bool, GenerationError> {
    text.push_str(&output.text);
    if let Some(sender) = sender
        && !output.text.is_empty()
    {
        send_event(sender, GenerationEvent::Content(output.text), control).await?;
    }
    Ok(output.matched)
}

async fn send_terminal(
    sender: &mpsc::Sender<GenerationEvent>,
    params: &crate::CompletionParams,
    completion: Completion,
    control: &GenerationControl,
) -> Result<(), GenerationError> {
    send_event(
        sender,
        GenerationEvent::Finished(completion.finish_reason, completion.usage),
        control,
    )
    .await?;
    if params
        .stream_options
        .is_some_and(|options| options.include_usage)
    {
        send_event(sender, GenerationEvent::Usage(completion.usage), control).await?;
    }
    send_event(sender, GenerationEvent::Done, control).await
}

async fn send_event(
    sender: &mpsc::Sender<GenerationEvent>,
    event: GenerationEvent,
    control: &GenerationControl,
) -> Result<(), GenerationError> {
    if control.is_cancelled() {
        return Err(GenerationError::cancelled());
    }
    let send = sender.send(event);
    let cancelled = control.cancellation.notify.notified();
    let stalled = tokio::time::sleep(control.write_stall_timeout);
    let mut send = std::pin::pin!(send);
    let mut cancelled = std::pin::pin!(cancelled);
    let mut stalled = std::pin::pin!(stalled);
    poll_fn(|context| {
        if let std::task::Poll::Ready(result) = send.as_mut().poll(context) {
            return std::task::Poll::Ready(result.map_err(|_| {
                control.cancel();
                GenerationError::cancelled()
            }));
        }
        if cancelled.as_mut().poll(context).is_ready() {
            return std::task::Poll::Ready(Err(GenerationError::cancelled()));
        }
        if stalled.as_mut().poll(context).is_ready() {
            control.cancel();
            return std::task::Poll::Ready(Err(GenerationError::cancelled()));
        }
        std::task::Poll::Pending
    })
    .await
}

fn read_token(bytes: &[u8]) -> Result<u32, GenerationError> {
    let bytes: [u8; 4] = bytes
        .try_into()
        .map_err(|_| GenerationError::new("selected token is not one u32"))?;
    Ok(u32::from_le_bytes(bytes))
}

async fn finalize_sequence<B>(
    runner: &mut EngineRunner<B>,
    pending: &mut VecDeque<EngineDecodeOutput>,
) -> Result<(), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let mut failure = None;
    for output in pending.drain(..) {
        if let Err(error) = runner.discard_queued_decode(output).await
            && failure.is_none()
        {
            failure = Some(error);
        }
    }
    failure.map_or(Ok(()), |error| {
        Err(GenerationError::unavailable(format!(
            "speculative decode cleanup failed: {error:?}"
        )))
    })
}

async fn prepare_chat_prompt<R>(
    request: &ChatCompletionRequest,
    config: &HostConfig,
    chat_info: &crate::ChatInfo,
    max_context: u32,
    vocab: u32,
    prompt: Arc<StdMutex<PromptArtifacts<R>>>,
    control: &GenerationControl,
) -> Result<(Vec<u32>, u32, ResolvedParams), GenerationError>
where
    R: PromptRenderer + 'static,
{
    let params = validate_chat_input(request, config, chat_info, vocab).map_err(|error| {
        if error.is_too_large() {
            GenerationError::too_large(error.to_string())
        } else {
            GenerationError::invalid(error.to_string())
        }
    })?;
    if control.is_cancelled() {
        return Err(GenerationError::cancelled());
    }
    let messages = serde_json::to_string(&request.messages)
        .map_err(|error| GenerationError::new(format!("cannot encode messages: {error}")))?;
    let tools = serde_json::to_string(&request.params.tools)
        .map_err(|error| GenerationError::new(format!("cannot encode tools: {error}")))?;
    let options = serde_json::to_string(&serde_json::json!({
        "add_generation_prompt": true,
        "tool_choice": request.params.tool_choice.as_ref(),
        "parallel_tool_calls": request.params.parallel_tool_calls,
        "response_format": request.params.response_format.as_ref(),
    }))
    .map_err(|error| GenerationError::new(format!("cannot encode render options: {error}")))?;
    let max_rendered_bytes = config.request.max_text_bytes.get();
    let tokens = tokio::task::spawn_blocking(move || {
        let mut prompt = prompt
            .lock()
            .map_err(|_| GenerationError::new("prompt processor lock is poisoned"))?;
        let prompt_text = prompt.renderer.render_prompt(&messages, &tools, &options)?;
        let rendered_bytes = u64::try_from(prompt_text.len())
            .map_err(|_| GenerationError::invalid("rendered prompt size exceeds u64"))?;
        if rendered_bytes > max_rendered_bytes {
            return Err(GenerationError::invalid(
                "rendered prompt exceeds the configured byte ceiling",
            ));
        }
        prompt
            .tokenizer
            .encode(prompt_text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| GenerationError::new(format!("cannot tokenize prompt: {error}")))
    })
    .await
    .map_err(|error| GenerationError::new(format!("prompt worker failed: {error}")))??;
    finish_prompt(tokens, params, max_context, control, "rendered prompt")
}

async fn prepare_completion_prompt<R>(
    request: &CompletionRequest,
    config: &HostConfig,
    chat_info: &crate::ChatInfo,
    max_context: u32,
    vocab: u32,
    prompt: Arc<StdMutex<PromptArtifacts<R>>>,
    control: &GenerationControl,
) -> Result<(Vec<u32>, u32, ResolvedParams), GenerationError>
where
    R: PromptRenderer + 'static,
{
    let params = validate_completion_input(request, config, chat_info, vocab)
        .map_err(|error| GenerationError::invalid(error.to_string()))?;
    if control.is_cancelled() {
        return Err(GenerationError::cancelled());
    }
    let text = request.prompt.clone();
    let tokens = tokio::task::spawn_blocking(move || {
        prompt
            .lock()
            .map_err(|_| GenerationError::new("prompt processor lock is poisoned"))?
            .tokenizer
            .encode(text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| GenerationError::invalid(format!("cannot tokenize prompt: {error}")))
    })
    .await
    .map_err(|error| GenerationError::new(format!("prompt worker failed: {error}")))??;
    finish_prompt(tokens, params, max_context, control, "prompt")
}

fn finish_prompt(
    tokens: Vec<u32>,
    params: ResolvedParams,
    max_context: u32,
    control: &GenerationControl,
    description: &str,
) -> Result<(Vec<u32>, u32, ResolvedParams), GenerationError> {
    if control.is_cancelled() {
        return Err(GenerationError::cancelled());
    }
    let prompt_tokens = u32::try_from(tokens.len())
        .map_err(|_| GenerationError::invalid("prompt token count exceeds u32"))?;
    if tokens.is_empty() {
        return Err(GenerationError::invalid(format!(
            "{description} tokenized to an empty sequence"
        )));
    }
    let params = validate_prompt(params, prompt_tokens, max_context)
        .map_err(|error| GenerationError::invalid(error.to_string()))?;
    Ok((tokens, prompt_tokens, params))
}

async fn sampling_seed(requested: Option<u64>, temperature: f32) -> Result<u64, GenerationError> {
    if let Some(seed) = requested {
        return Ok(seed);
    }
    if temperature == 0.0 {
        return Ok(0);
    }
    tokio::task::spawn_blocking(random_seed)
        .await
        .map_err(|error| GenerationError::new(format!("sampling worker failed: {error}")))?
}

fn random_seed() -> Result<u64, GenerationError> {
    let mut bytes = [0_u8; 8];
    File::open("/dev/urandom")
        .and_then(|mut file| file.read_exact(&mut bytes))
        .map_err(|error| GenerationError::new(format!("cannot read sampling entropy: {error}")))?;
    Ok(u64::from_ne_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, atomic::AtomicUsize};

    struct CountingRenderer(Arc<AtomicUsize>);

    impl PromptRenderer for CountingRenderer {
        fn render_prompt(&mut self, _: &str, _: &str, _: &str) -> Result<String, GenerationError> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok("rendered".to_owned())
        }
    }

    #[tokio::test]
    async fn invalid_message_count_never_calls_chat_format() {
        let calls = Arc::new(AtomicUsize::new(0));
        let prompt = Arc::new(StdMutex::new(PromptArtifacts {
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            renderer: CountingRenderer(Arc::clone(&calls)),
        }));
        let mut config = HostConfig::default();
        config.request.max_messages = 0;
        let request = ChatCompletionRequest {
            model: "test".to_owned(),
            messages: vec![crate::Message {
                role: "user".to_owned(),
                content: Some("hello".to_owned()),
                tool_calls: None,
                tool_call_id: None,
            }],
            params: crate::CompletionParams::default(),
        };
        let info = crate::ChatInfo {
            stop_token_ids: Vec::new(),
            sampling: crate::SamplingDefaults::default(),
        };
        let control = GenerationControl::new(Duration::from_secs(1));

        let result =
            prepare_chat_prompt(&request, &config, &info, 32_768, 151_936, prompt, &control).await;

        assert!(result.is_err());
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn greedy_sampling_needs_no_entropy() {
        assert_eq!(sampling_seed(None, 0.0).await.unwrap(), 0);
        assert_eq!(sampling_seed(Some(17), 0.7).await.unwrap(), 17);
    }

    #[tokio::test]
    async fn write_stall_cancels_idempotently() {
        let (sender, _receiver) = mpsc::channel(1);
        sender.try_send(GenerationEvent::Role).unwrap();
        let control = GenerationControl::new(Duration::from_millis(1));
        assert!(
            send_event(&sender, GenerationEvent::Done, &control)
                .await
                .is_err()
        );
        assert!(control.is_cancelled());
        control.cancel();
        assert!(control.is_cancelled());
    }
}
