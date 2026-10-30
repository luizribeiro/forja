use std::{
    collections::VecDeque,
    error::Error,
    fmt,
    fs::File,
    future::{Future, poll_fn},
    io::Read,
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use forja_config::{Choice, HostConfig, Unbounded};
use forja_core::{Backend, correction_sampling_seed};
use forja_host::{
    BatchPhase, BatchSelection, EngineBeginDecode, EngineDecode, EngineDecodeOutput, EngineInfo,
    EngineRunner, EngineTensor, SamplingParams,
};
use tokenizers::Tokenizer;
use tokio::sync::{Notify, mpsc};

use crate::guidance::{GuidanceSession, GuidanceWorker};
use crate::metrics::{ServingMetrics, observe, observe_phase};
use crate::prefix::{PrefixIdentity, PrefixIndex, PrefixLedger, PrefixLimits, PrefixNamespace};
use crate::request::{ToolChoice, validate_chat_input, validate_completion_input, validate_prompt};
use crate::tool_output::{GeneratedToolCall, ToolDelta, ToolOutput};
use crate::{ChatCompletionRequest, ChatFormat, CompletionRequest, ResolvedParams, StopMatcher};

mod continuous;

static NEXT_CHAT_STREAM: AtomicU64 = AtomicU64::new(1);

/// A failure while preparing or running a generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GenerationError {
    message: String,
    kind: ErrorKind,
    paged_bank_in_flight: bool,
}

impl GenerationError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::Internal,
            paged_bank_in_flight: false,
        }
    }

    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::Invalid,
            paged_bank_in_flight: false,
        }
    }

    pub(crate) fn too_large(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::PayloadTooLarge,
            paged_bank_in_flight: false,
        }
    }

    pub(crate) fn context_length(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::ContextLength,
            paged_bank_in_flight: false,
        }
    }

    pub(crate) fn unavailable(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            kind: ErrorKind::Unavailable,
            paged_bank_in_flight: false,
        }
    }

    fn paged_admission(error: &forja_host::bindings::l9o::gpu::compute::Error) -> Self {
        Self {
            message: format!("paged admission failed: {error:?}"),
            kind: ErrorKind::Unavailable,
            paged_bank_in_flight: forja_host::is_paged_bank_in_flight(error),
        }
    }

    fn cancelled() -> Self {
        Self {
            message: "generation cancelled".to_owned(),
            kind: ErrorKind::Unavailable,
            paged_bank_in_flight: false,
        }
    }

    fn paged_bank_in_flight(&self) -> bool {
        self.paged_bank_in_flight
    }

    pub(crate) const fn kind(&self) -> ErrorKind {
        self.kind
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ErrorKind {
    Invalid,
    ContextLength,
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
    cached_tokens: u32,
    started: Instant,
    /// Engine sampling parameters used by every decode step.
    pub sampling: SamplingParams,
    ledger: PrefixLedger,
    paged: bool,
    constraint: Option<ConstraintPlan>,
}

struct ConstraintPlan {
    grammar: String,
    trigger: String,
    trigger_tokens: Vec<u32>,
    forced: bool,
    tools: Vec<crate::Tool>,
    selected: Option<String>,
}

struct ToolParser {
    stream: u64,
    trigger: String,
    prompt: Arc<StdMutex<PromptArtifacts<ChatFormat>>>,
    output: ToolOutput,
}

impl ToolParser {
    fn new(
        plan: &ConstraintPlan,
        prompt: Arc<StdMutex<PromptArtifacts<ChatFormat>>>,
    ) -> Result<Self, GenerationError> {
        let stream = NEXT_CHAT_STREAM
            .try_update(Ordering::AcqRel, Ordering::Acquire, |stream| {
                stream.checked_add(1)
            })
            .map_err(|_| GenerationError::new("chat parser stream id space is exhausted"))?;
        Ok(Self {
            stream,
            trigger: plan.trigger.clone(),
            prompt,
            output: ToolOutput::new(&plan.tools, plan.selected.clone()),
        })
    }

    async fn seed(&mut self) -> Result<Vec<ToolDelta>, GenerationError> {
        let trigger = self.trigger.clone();
        self.parse(trigger).await
    }

    async fn parse(&mut self, chunk: String) -> Result<Vec<ToolDelta>, GenerationError> {
        let prompt = Arc::clone(&self.prompt);
        let stream = self.stream;
        let events = tokio::task::spawn_blocking(move || {
            prompt
                .lock()
                .map_err(|_| GenerationError::new("prompt processor lock is poisoned"))?
                .renderer
                .parse(stream, &chunk)
                .map_err(|error| GenerationError::new(format!("chat format failed: {error}")))?
                .map_err(|error| {
                    GenerationError::new(format!("chat format parser failed: {error}"))
                })
        })
        .await
        .map_err(|error| GenerationError::new(format!("chat parser worker failed: {error}")))??;
        self.output.apply(events).map_err(|error| {
            GenerationError::new(format!("chat format event invariant failed: {error}"))
        })
    }

    fn finish(&self) -> Result<GeneratedToolCall, GenerationError> {
        self.output.finish().map_err(|error| {
            GenerationError::new(format!("chat format event invariant failed: {error}"))
        })
    }

    async fn reset(self) {
        let prompt = self.prompt;
        let stream = self.stream;
        let _ =
            tokio::task::spawn_blocking(move || prompt.lock().ok()?.renderer.reset(stream).ok())
                .await;
    }
}

async fn start_tool_parser(
    plan: Option<&ConstraintPlan>,
    prompt: Arc<StdMutex<PromptArtifacts<ChatFormat>>>,
    forced: bool,
    sender: Option<&mpsc::Sender<GenerationEvent>>,
    control: &GenerationControl,
) -> Result<Option<ToolParser>, GenerationError> {
    let Some(plan) = plan else {
        return Ok(None);
    };
    let mut parser = ToolParser::new(plan, prompt)?;
    if forced {
        let deltas = parser.seed().await?;
        emit_tool_deltas(deltas, sender, control).await?;
    }
    Ok(Some(parser))
}

async fn emit_tool_deltas(
    deltas: Vec<ToolDelta>,
    sender: Option<&mpsc::Sender<GenerationEvent>>,
    control: &GenerationControl,
) -> Result<(), GenerationError> {
    let Some(sender) = sender else {
        return Ok(());
    };
    for delta in deltas {
        let event = match delta {
            ToolDelta::Start(name) => GenerationEvent::ToolCallStart(name),
            ToolDelta::Arguments(arguments) => GenerationEvent::ToolArguments(arguments),
        };
        send_event(sender, event, control).await?;
    }
    Ok(())
}

async fn parse_tool_piece(
    parser: &mut Option<ToolParser>,
    piece: Option<String>,
    sender: Option<&mpsc::Sender<GenerationEvent>>,
    control: &GenerationControl,
) -> Result<(), GenerationError> {
    if let (Some(parser), Some(piece)) = (parser, piece) {
        let deltas = parser.parse(piece).await?;
        emit_tool_deltas(deltas, sender, control).await?;
    }
    Ok(())
}

async fn seed_tool_parser(
    parser: &mut Option<ToolParser>,
    sender: Option<&mpsc::Sender<GenerationEvent>>,
    control: &GenerationControl,
) -> Result<(), GenerationError> {
    let deltas = parser
        .as_mut()
        .ok_or_else(|| GenerationError::new("tool parser is missing"))?
        .seed()
        .await?;
    emit_tool_deltas(deltas, sender, control).await
}

async fn finish_tool_parser(
    generation: &Result<FinishReason, GenerationError>,
    parser: Option<ToolParser>,
) -> Result<Option<GeneratedToolCall>, GenerationError> {
    let call = if matches!(generation, Ok(FinishReason::ToolCalls)) {
        parser
            .as_ref()
            .ok_or_else(|| GenerationError::new("tool parser is missing"))?
            .finish()
            .map(Some)
    } else {
        Ok(None)
    };
    if let Some(parser) = parser {
        parser.reset().await;
    }
    call
}

trait PromptRenderer: Send {
    fn render_prompt(
        &mut self,
        messages: &str,
        tools: &str,
        options: &str,
    ) -> Result<String, GenerationError>;

    fn grammar(
        &mut self,
        tools: &str,
        choice: &str,
        response: &str,
    ) -> Result<crate::GrammarOutput, GenerationError> {
        let _ = (tools, choice, response);
        Err(GenerationError::new("tool grammar is unavailable"))
    }
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

    fn grammar(
        &mut self,
        tools: &str,
        choice: &str,
        response: &str,
    ) -> Result<crate::GrammarOutput, GenerationError> {
        self.grammar(tools, choice, response)
            .map_err(|error| GenerationError::new(format!("chat format failed: {error}")))?
            .map_err(|error| GenerationError::new(format!("chat format grammar failed: {error}")))
    }
}

/// Why token generation ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FinishReason {
    /// A model-declared token or request stop string matched.
    Stop,
    /// The validated output-token limit was reached.
    Length,
    /// A complete function call was generated.
    ToolCalls,
}

/// Token counts for one completed request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Usage {
    /// Tokens in the rendered prompt.
    pub prompt_tokens: u32,
    /// Prompt tokens served from a retained paged KV prefix.
    pub cached_tokens: u32,
    /// Sampled tokens, including a token that caused a stop.
    pub completion_tokens: u32,
}

/// Fully decoded output from one request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Completion {
    /// Decoded text before any matched stop.
    pub text: String,
    /// A validated function call, when constrained generation completed one.
    pub tool_call: Option<GeneratedToolCall>,
    /// Why generation ended.
    pub finish_reason: FinishReason,
    /// Request token counts.
    pub usage: Usage,
    pub(crate) token_ids: Vec<u32>,
}

impl Completion {
    /// Returns the selected completion token ids.
    #[must_use]
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }
}

/// One logical event produced during a streamed generation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GenerationEvent {
    /// Initial assistant role for a chat stream.
    Role,
    /// Decoded content safe to expose to the client.
    Content(String),
    /// Begins the only function call in a streamed turn.
    ToolCallStart(String),
    /// Appends canonical JSON to the streamed function arguments.
    ToolArguments(String),
    /// The terminal choice chunk.
    Finished(FinishReason, Usage),
    /// Optional usage-only chunk.
    Usage(Usage),
    /// Clean stream terminator.
    Done,
    /// One terminal error emitted after response headers.
    Error(GenerationError),
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
    guidance: GuidanceWorker,
    prefix_namespace: PrefixNamespace,
    prefix_index: Option<PrefixIndex>,
    metrics: ServingMetrics,
    config_generation: u64,
    scheduler: SchedulerConfig,
    #[cfg(test)]
    batch_test_barrier: Option<Arc<BatchTestBarrier>>,
}

impl<B: Backend + Send + Sync + 'static> ServingEngine<B> {
    #[cfg(test)]
    fn trace_request(&self, id: crate::scheduler::RequestId, first_token: bool) {
        if let Some(barrier) = &self.batch_test_barrier {
            barrier.note_request(id, first_token);
        }
    }

    #[cfg(test)]
    fn trace_prompt_work(&self, budget: u32, decoding: bool) {
        if let Some(barrier) = &self.batch_test_barrier {
            barrier.note_prompt(budget, decoding);
        }
    }

    #[cfg(test)]
    fn trace_prompt_round(&self, decoding: bool) {
        if !decoding && let Some(barrier) = &self.batch_test_barrier {
            barrier.idle_rounds.fetch_add(1, Ordering::AcqRel);
        }
    }

    pub(crate) fn wait_for_quiescence(
        &self,
    ) -> impl Future<Output = Result<(), forja_core::BackendError>> + Send + 'static {
        self.runner.wait_for_queued_submissions()
    }
}

#[cfg(test)]
pub(crate) struct BatchTestBarrier {
    armed: AtomicBool,
    admission_overlapped: AtomicBool,
    retirement_drained: AtomicBool,
    prefill_overlap_width: AtomicU64,
    pause_prefill: AtomicBool,
    decoded_batches: AtomicU64,
    idle_rounds: AtomicU64,
    prompt_pieces: StdMutex<Vec<usize>>,
    prompt_trace: StdMutex<Vec<(u32, bool, u64)>>,
    request_trace: StdMutex<Vec<(crate::scheduler::RequestId, bool, u64)>>,
    deferred_fallthroughs: AtomicU64,
    reached: Notify,
    resume: Notify,
    finish_sequence: AtomicU64,
    ordinary_finished: AtomicU64,
    constrained_finished: AtomicU64,
    cancel_on_batch: StdMutex<Option<GenerationControl>>,
    refuse_release: AtomicBool,
    refuse_bind: AtomicBool,
}

#[cfg(test)]
impl BatchTestBarrier {
    pub(crate) const fn new() -> Self {
        Self::with_armed(true)
    }

    pub(crate) const fn passive() -> Self {
        Self::with_armed(false)
    }

    const fn with_armed(armed: bool) -> Self {
        Self {
            armed: AtomicBool::new(armed),
            admission_overlapped: AtomicBool::new(false),
            retirement_drained: AtomicBool::new(false),
            prefill_overlap_width: AtomicU64::new(0),
            pause_prefill: AtomicBool::new(false),
            decoded_batches: AtomicU64::new(0),
            idle_rounds: AtomicU64::new(0),
            prompt_pieces: StdMutex::new(Vec::new()),
            prompt_trace: StdMutex::new(Vec::new()),
            request_trace: StdMutex::new(Vec::new()),
            deferred_fallthroughs: AtomicU64::new(0),
            reached: Notify::const_new(),
            resume: Notify::const_new(),
            finish_sequence: AtomicU64::new(0),
            ordinary_finished: AtomicU64::new(0),
            constrained_finished: AtomicU64::new(0),
            cancel_on_batch: StdMutex::new(None),
            refuse_release: AtomicBool::new(false),
            refuse_bind: AtomicBool::new(false),
        }
    }

    pub(crate) fn prefill_race() -> Self {
        let barrier = Self::new();
        barrier.pause_prefill.store(true, Ordering::Release);
        barrier
    }

    pub(crate) fn prompt_trace(&self) -> Vec<(u32, bool, u64)> {
        self.prompt_trace.lock().unwrap().clone()
    }

    pub(crate) fn prompt_pieces(&self) -> Vec<usize> {
        self.prompt_pieces.lock().unwrap().clone()
    }

    pub(crate) fn request_trace(&self) -> Vec<(crate::scheduler::RequestId, bool, u64)> {
        self.request_trace.lock().unwrap().clone()
    }

    pub(crate) fn idle_rounds(&self) -> u64 {
        self.idle_rounds.load(Ordering::Acquire)
    }

    fn note_request(&self, id: crate::scheduler::RequestId, first_token: bool) {
        self.request_trace
            .lock()
            .unwrap()
            .push((id, first_token, self.idle_rounds()));
    }

    fn note_prompt(&self, budget: u32, decoding: bool) {
        self.prompt_trace.lock().unwrap().push((
            budget,
            decoding,
            self.decoded_batches.load(Ordering::Acquire),
        ));
    }

    async fn pause_prefill_once(&self) {
        if self.pause_prefill.swap(false, Ordering::AcqRel) {
            self.pause_once().await;
        }
    }

    pub(crate) fn cancel_on_next_batch(&self, control: GenerationControl) {
        if let Ok(mut cancel) = self.cancel_on_batch.lock() {
            *cancel = Some(control);
        }
        self.refuse_release.store(true, Ordering::Release);
    }

    pub(crate) fn stale_lane() -> Self {
        let barrier = Self::passive();
        barrier.refuse_bind.store(true, Ordering::Release);
        barrier
    }

    pub(crate) async fn reached(&self) {
        self.reached.notified().await;
    }

    pub(crate) fn release(&self) {
        self.resume.notify_one();
    }

    pub(crate) fn admission_overlapped(&self) -> bool {
        self.admission_overlapped.load(Ordering::Acquire)
    }

    pub(crate) fn retirement_drained(&self) -> bool {
        self.retirement_drained.load(Ordering::Acquire)
    }

    pub(crate) fn prefill_overlap_width(&self) -> u64 {
        self.prefill_overlap_width.load(Ordering::Acquire)
    }

    pub(crate) fn deferred_fallthroughs(&self) -> u64 {
        self.deferred_fallthroughs.load(Ordering::Acquire)
    }

    pub(crate) fn finish_order(&self) -> (u64, u64) {
        (
            self.ordinary_finished.load(Ordering::Acquire),
            self.constrained_finished.load(Ordering::Acquire),
        )
    }

    fn note_finish(&self, constrained: bool) {
        let sequence = self.finish_sequence.fetch_add(1, Ordering::AcqRel) + 1;
        if constrained {
            self.constrained_finished.store(sequence, Ordering::Release);
        } else {
            self.ordinary_finished.store(sequence, Ordering::Release);
        }
    }

    fn note_admission_overlap(&self) {
        self.admission_overlapped.store(true, Ordering::Release);
    }

    fn note_retirement_drain(&self) {
        self.retirement_drained.store(true, Ordering::Release);
    }

    fn note_prefill_overlap(&self, width: usize) {
        self.prefill_overlap_width
            .fetch_max(u64::try_from(width).unwrap_or(u64::MAX), Ordering::AcqRel);
    }

    fn note_deferred_fallthrough(&self) {
        self.deferred_fallthroughs.fetch_add(1, Ordering::AcqRel);
    }

    fn cancel_lane(&self) {
        if let Ok(mut cancel) = self.cancel_on_batch.lock()
            && let Some(control) = cancel.take()
        {
            control.cancel();
        }
    }

    fn release_is_blocked(&self) -> bool {
        self.refuse_release.load(Ordering::Acquire)
    }

    fn observe_batch_completion(&self) {
        self.refuse_release.store(false, Ordering::Release);
    }

    fn refuse_bind_once(&self) -> bool {
        self.refuse_bind.swap(false, Ordering::AcqRel)
    }

    async fn pause_once(&self) {
        if self.armed.swap(false, Ordering::AcqRel) {
            self.reached.notify_one();
            self.resume.notified().await;
        }
    }
}

#[derive(Debug)]
struct SchedulerConfig {
    active_lanes: u32,
    max_prefill_chunks: u32,
    prompt_token_budget: u32,
    profile_prompt_budget: u32,
    profile_prefill_chunk: u32,
    idle_prompt_token_budget: u32,
    prefill_chunk_multiple: Option<std::num::NonZeroU32>,
    decode_buckets: Vec<u32>,
}

fn validate_prompt_block(
    info: &EngineInfo,
    config: &HostConfig,
    profile_budget: u32,
    profile_chunk: u32,
) -> Result<(), GenerationError> {
    if info.prompt_block_tokens.is_none() {
        return Ok(());
    }
    let prompt = resolve_u64_ceiling(
        config.serving.prompt_token_budget,
        u64::from(profile_budget),
        "serving.prompt_token_budget",
    )?;
    let idle = resolve_u64_ceiling(
        config.serving.idle_prompt_token_budget,
        u64::from(profile_chunk),
        "serving.idle_prompt_token_budget",
    )?;
    validate_resolved_prompt_block(info, prompt, idle, profile_chunk, info.prompt_block_tokens)
}

fn validate_resolved_prompt_block(
    info: &EngineInfo,
    prompt: u64,
    idle: u64,
    chunk: u32,
    multiple: Option<std::num::NonZeroU32>,
) -> Result<(), GenerationError> {
    let Some(block) = info.prompt_block_tokens.map(std::num::NonZeroU32::get) else {
        return Ok(());
    };
    if prompt < u64::from(block)
        || idle < u64::from(block)
        || !chunk.is_multiple_of(block)
        || multiple.is_none_or(|multiple| !multiple.get().is_multiple_of(block))
    {
        return Err(GenerationError::new(
            "prompt budgets must admit the engine's atomic numerical block",
        ));
    }
    Ok(())
}

impl SchedulerConfig {
    fn resolve(
        info: &EngineInfo,
        config: &HostConfig,
        profile_budget: u32,
        profile_prefill_chunk: u32,
    ) -> Result<Self, GenerationError> {
        validate_prompt_block(info, config, profile_budget, profile_prefill_chunk)?;
        let Some(serving) = info.serving.as_ref() else {
            let atomic = info
                .prompt_block_tokens
                .map_or(1, std::num::NonZeroU32::get);
            return Ok(Self {
                active_lanes: 1,
                max_prefill_chunks: 1,
                prompt_token_budget: atomic,
                profile_prompt_budget: profile_budget,
                profile_prefill_chunk,
                idle_prompt_token_budget: atomic,
                prefill_chunk_multiple: info.prompt_block_tokens,
                decode_buckets: vec![1],
            });
        };
        if profile_budget == 0 || profile_prefill_chunk == 0 {
            return Err(GenerationError::new(
                "profile prompt limits must be positive",
            ));
        }
        let configured_active = resolve_u32_ceiling(
            config.serving.max_active_sequences,
            serving.active_lanes,
            "serving.max_active_sequences",
        )?;
        let configured_batch = resolve_u32_ceiling(
            config.serving.max_batch_size,
            configured_active,
            "serving.max_batch_size",
        )?;
        let max_prefill_chunks = resolve_u32_ceiling(
            config.serving.max_prefill_chunks,
            configured_active,
            "serving.max_prefill_chunks",
        )?;
        let configured_prompt = resolve_u64_ceiling(
            config.serving.prompt_token_budget,
            u64::from(profile_budget),
            "serving.prompt_token_budget",
        )?;
        let configured_lanes = configured_active.min(configured_batch);
        let declared_buckets = serving
            .batch_shapes
            .iter()
            .filter(|shape| {
                shape.phase == BatchPhase::Decode && shape.selection == BatchSelection::Combined
            })
            .map(|shape| shape.bucket)
            .collect::<Vec<_>>();
        let (active_lanes, decode_buckets) =
            scheduler_decode_buckets(declared_buckets, configured_lanes);
        let largest_prompt_chunk = largest_prompt_chunk(serving, profile_prefill_chunk);
        let prompt_token_budget = profile_budget
            .min(u32::try_from(configured_prompt).unwrap_or(u32::MAX))
            .min(largest_prompt_chunk);
        let idle_prompt_token_budget = u32::try_from(resolve_u64_ceiling(
            config.serving.idle_prompt_token_budget,
            u64::from(profile_prefill_chunk),
            "serving.idle_prompt_token_budget",
        )?)
        .unwrap_or(u32::MAX)
        .min(largest_prompt_chunk);
        let prefill_chunk_multiple = serving
            .prefill_chunk_multiple
            .or(info.prompt_block_tokens)
            .filter(|multiple| {
                prompt_token_budget.is_multiple_of(multiple.get())
                    && idle_prompt_token_budget >= multiple.get()
            });
        validate_resolved_prompt_block(
            info,
            u64::from(prompt_token_budget),
            u64::from(idle_prompt_token_budget),
            profile_prefill_chunk,
            prefill_chunk_multiple,
        )?;
        if !decode_buckets.is_empty() {
            crate::scheduler::SchedulerPolicy::new(
                decode_buckets.clone(),
                usize::try_from(active_lanes).unwrap_or(usize::MAX),
                usize::try_from(max_prefill_chunks).unwrap_or(usize::MAX),
                prompt_token_budget,
                idle_prompt_token_budget,
            )
            .map_err(GenerationError::new)?;
        }
        Ok(Self {
            active_lanes,
            max_prefill_chunks,
            prompt_token_budget,
            profile_prompt_budget: profile_budget,
            profile_prefill_chunk,
            idle_prompt_token_budget,
            prefill_chunk_multiple,
            decode_buckets,
        })
    }
}

fn largest_prompt_chunk(serving: &forja_host::ServingInfo, fallback: u32) -> u32 {
    serving
        .batch_shapes
        .iter()
        .filter(|shape| {
            shape.phase == BatchPhase::Prompt
                && shape.bucket == 1
                && shape.selection == BatchSelection::Combined
        })
        .map(|shape| shape.tokens)
        .max()
        .unwrap_or(fallback)
}

fn resolve_u32_ceiling(
    configured: Choice<std::num::NonZeroU32>,
    declared: u32,
    field: &str,
) -> Result<u32, GenerationError> {
    match configured {
        Choice::Auto => Ok(declared),
        Choice::Fixed(value) if value.get() <= declared => Ok(value.get()),
        Choice::Fixed(_) => Err(GenerationError::new(format!(
            "{field} exceeds the engine-declared ceiling of {declared}"
        ))),
    }
}

fn resolve_u64_ceiling(
    configured: Choice<std::num::NonZeroU64>,
    declared: u64,
    field: &str,
) -> Result<u64, GenerationError> {
    match configured {
        Choice::Auto => Ok(declared),
        Choice::Fixed(value) if value.get() <= declared => Ok(value.get()),
        Choice::Fixed(_) => Err(GenerationError::new(format!(
            "{field} exceeds the engine-declared ceiling of {declared}"
        ))),
    }
}

fn prefix_limits(
    serving: &forja_host::ServingInfo,
    config: &HostConfig,
    max_context: u32,
) -> Result<PrefixLimits, GenerationError> {
    let declared_entries = config.server.queue_length.max(serving.active_lanes);
    let declared_tokens = u64::from(declared_entries)
        .checked_mul(u64::from(max_context))
        .ok_or_else(|| GenerationError::new("prefix token limit overflowed"))?;
    let pages_per_entry = max_context.div_ceil(serving.page_tokens);
    let declared_pages = declared_entries
        .checked_mul(pages_per_entry)
        .ok_or_else(|| GenerationError::new("prefix page limit overflowed"))?;
    let entries = resolve_u32_ceiling(
        config.serving.prefix_cache_entries,
        declared_entries,
        "serving.prefix_cache_entries",
    )?;
    let tokens = resolve_u64_ceiling(
        config.serving.prefix_cache_tokens,
        declared_tokens,
        "serving.prefix_cache_tokens",
    )?;
    let pages = resolve_u32_ceiling(
        config.serving.prefix_cache_pages,
        declared_pages,
        "serving.prefix_cache_pages",
    )?;
    Ok(PrefixLimits {
        entries: usize::try_from(entries)
            .map_err(|_| GenerationError::new("prefix entry limit overflowed"))?,
        tokens: usize::try_from(tokens)
            .map_err(|_| GenerationError::new("prefix token limit overflowed"))?,
        referenced_pages: usize::try_from(pages)
            .map_err(|_| GenerationError::new("prefix page limit overflowed"))?,
    })
}

fn scheduler_decode_buckets(mut buckets: Vec<u32>, configured_lanes: u32) -> (u32, Vec<u32>) {
    buckets.sort_unstable();
    buckets.dedup();
    if buckets.is_empty() {
        return (configured_lanes, buckets);
    }
    let active_lanes = configured_lanes.min(buckets.last().copied().unwrap_or(0));
    let successor = buckets
        .iter()
        .copied()
        .find(|&bucket| bucket >= active_lanes);
    buckets.retain(|&bucket| bucket <= active_lanes || Some(bucket) == successor);
    (active_lanes, buckets)
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: String,
        runner: EngineRunner<B>,
        tokenizer: Tokenizer,
        chat_format: ChatFormat,
        config: HostConfig,
        info: EngineInfo,
        max_context: u32,
        profile_prompt_budget: u32,
        profile_prefill_chunk: u32,
        prefix_identity: PrefixIdentity,
    ) -> Result<Self, GenerationError> {
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
        let guidance =
            GuidanceWorker::new(&tokenizer, info.vocab, chat_info.stop_token_ids.as_slice())
                .map_err(|error| {
                    GenerationError::new(format!("cannot initialize constraints: {error}"))
                })?;
        let prefix_index = info
            .serving
            .as_ref()
            .map(|serving| {
                let limits = prefix_limits(serving, &config, max_context)?;
                PrefixIndex::new(limits).map_err(|error| GenerationError::new(error.to_string()))
            })
            .transpose()?;
        let scheduler =
            SchedulerConfig::resolve(&info, &config, profile_prompt_budget, profile_prefill_chunk)?;
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
            guidance,
            prefix_namespace: prefix_identity.into_namespace(),
            prefix_index,
            metrics: ServingMetrics::default(),
            config_generation: 0,
            scheduler,
            #[cfg(test)]
            batch_test_barrier: None,
        })
    }

    #[cfg(test)]
    pub(crate) fn set_batch_test_barrier(&mut self, barrier: Arc<BatchTestBarrier>) {
        self.batch_test_barrier = Some(barrier);
    }

    /// Returns the only model identifier accepted by this engine.
    #[must_use]
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Returns bounded aggregate serving measurements.
    #[must_use]
    pub const fn metrics(&self) -> &ServingMetrics {
        &self.metrics
    }

    pub(crate) fn host_config(&self) -> HostConfig {
        self.config.clone()
    }

    pub(crate) fn update_policy(
        &mut self,
        generation: u64,
        config: HostConfig,
    ) -> Result<(), GenerationError> {
        let scheduler = SchedulerConfig::resolve(
            &self.info,
            &config,
            self.scheduler.profile_prompt_budget,
            self.scheduler.profile_prefill_chunk,
        )?;
        if let (Some(serving), Some(index)) = (&self.info.serving, &mut self.prefix_index) {
            let limits = prefix_limits(serving, &config, self.max_context)?;
            index
                .reconfigure(limits, config.serving.prefix_reuse)
                .map_err(|error| GenerationError::new(error.to_string()))?;
        }
        self.runner.update_time_limits(
            config.limits.guest_call_timeout.get(),
            config.limits.submission_timeout.get(),
            match config.limits.gpu_time_budget {
                Unbounded::Unlimited => Duration::MAX,
                Unbounded::Limited(duration) => duration.get(),
            },
        );
        self.scheduler = scheduler;
        self.config = config;
        self.config_generation = generation;
        Ok(())
    }

    pub(crate) const fn supports_continuous_batching(&self) -> bool {
        !self.scheduler.decode_buckets.is_empty()
    }

    fn refresh_runtime_metrics(&mut self) {
        let execution = self.runner.metrics();
        self.metrics.gpu_time_ns = u64::try_from(execution.gpu_time.as_nanos()).unwrap_or(u64::MAX);
        self.metrics.prepared_graphs = execution.prepared_graphs;
        self.metrics.graph_preparation_time_ns =
            u64::try_from(execution.graph_preparation_time.as_nanos()).unwrap_or(u64::MAX);
        self.metrics.live_bytes = execution.live_bytes;
        if let Ok(Some(memory)) = self.runner.paged_memory_usage() {
            self.metrics.pool_bytes = memory.slab_bytes;
            self.metrics.table_bytes = memory.table_bytes;
            self.metrics.slot_bytes = memory.slot_bytes;
            self.metrics.graph_bytes = memory.graph_bytes;
            self.metrics.scratch_bytes = memory.scratch_bytes;
        }
    }

    pub(crate) fn max_body_bytes(&self) -> usize {
        usize::try_from(self.config.request.max_body_bytes.get()).unwrap_or(usize::MAX)
    }

    pub(crate) const fn queue_length(&self) -> u32 {
        self.config.server.queue_length
    }

    pub(crate) fn declared_active_sequences(&self) -> u32 {
        self.info
            .serving
            .as_ref()
            .map_or(1, |serving| serving.active_lanes)
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
        let (tokens, prompt_tokens, params, constraint) = prepare_chat_prompt(
            request,
            &self.config,
            &self.chat_info,
            self.max_context,
            self.info.vocab,
            Arc::clone(&self.prompt),
            control,
        )
        .await?;
        self.prefill(tokens, prompt_tokens, params, constraint, control)
            .await
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
        self.prefill(tokens, prompt_tokens, params, None, control)
            .await
    }

    fn prepare_prefix(
        &mut self,
        tokens: &[u32],
        prompt_tokens: u32,
        max_tokens: u32,
    ) -> Result<(usize, bool), GenerationError> {
        let reserved_tokens = u64::from(prompt_tokens)
            .checked_add(u64::from(max_tokens))
            .ok_or_else(|| GenerationError::new("paged reservation overflowed"))?;
        let hit = if self.config.serving.prefix_reuse {
            self.prefix_index
                .as_mut()
                .map(|index| index.lookup(self.prefix_namespace, tokens, reserved_tokens))
                .transpose()
                .map_err(|error| GenerationError::unavailable(error.to_string()))?
                .flatten()
        } else {
            None
        };
        let indexed_len = hit.as_ref().map_or(0, |hit| hit.cached_tokens);
        let paged = self
            .runner
            .prepare_paged_request(hit.map(|hit| hit.sequence), reserved_tokens)
            .map_err(|error| {
                GenerationError::unavailable(format!("paged admission failed: {error:?}"))
            })?;
        if paged {
            Ok((indexed_len, true))
        } else {
            Ok((0, false))
        }
    }

    async fn prefill(
        &mut self,
        tokens: Vec<u32>,
        prompt_tokens: u32,
        params: ResolvedParams,
        constraint: Option<ConstraintPlan>,
        control: &GenerationControl,
    ) -> Result<PrefilledPrompt, GenerationError> {
        let started = Instant::now();
        let sampling = SamplingParams {
            temperature: params.temperature,
            top_k: params.top_k,
            top_p: params.top_p,
            seed: sampling_seed(params.seed, params.temperature).await?,
        };
        if control.is_cancelled() {
            return Err(GenerationError::cancelled());
        }
        let ledger = PrefixLedger::new(tokens.clone());
        let match_started = Instant::now();
        let (cached_len, paged) = self.prepare_prefix(&tokens, prompt_tokens, params.max_tokens)?;
        self.metrics.requests = self.metrics.requests.saturating_add(1);
        observe(&mut self.metrics.match_time, match_started.elapsed());
        if cached_len == 0 {
            self.metrics.misses = self.metrics.misses.saturating_add(1);
        } else {
            self.metrics.hits = self.metrics.hits.saturating_add(1);
            self.metrics.reused_tokens = self
                .metrics
                .reused_tokens
                .saturating_add(u64::try_from(cached_len).unwrap_or(u64::MAX));
        }
        let start_pos = u32::try_from(cached_len)
            .map_err(|_| GenerationError::new("cached prefix length overflowed"))?;
        let prefill_started = Instant::now();
        let first = self
            .runner
            .enqueue_decode(EngineDecode {
                phase: forja_host::BatchPhase::Prompt,
                tokens: Some(tokens[cached_len..].to_vec()),
                start_pos,
                sampling,
            })
            .await
            .map_err(|error| {
                GenerationError::unavailable(format!("engine prefill failed: {error}"))
            })
            .and_then(|result| {
                result.map_err(|error| {
                    GenerationError::unavailable(format!("engine prefill failed: {error:?}"))
                })
            });
        let first = match first {
            Ok(first) => first,
            Err(error) => return Err(error),
        };
        observe(
            &mut self.metrics.suffix_prefill_time,
            prefill_started.elapsed(),
        );
        Ok(PrefilledPrompt {
            first,
            params,
            prompt_tokens,
            cached_tokens: start_pos,
            started,
            sampling,
            ledger,
            paged,
            constraint,
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

    async fn start_decode(
        &mut self,
        prefill: PrefilledPrompt,
        sender: Option<&mpsc::Sender<GenerationEvent>>,
        control: &GenerationControl,
    ) -> Result<DecodeStart, GenerationError> {
        prepare_decode(
            &mut self.runner,
            &self.guidance,
            prefill,
            Arc::clone(&self.prompt),
            sender,
            control,
        )
        .await
    }

    #[allow(clippy::too_many_lines)]
    async fn decode(
        &mut self,
        prefill: PrefilledPrompt,
        sender: Option<&mpsc::Sender<GenerationEvent>>,
        control: &GenerationControl,
    ) -> Result<Completion, GenerationError> {
        let constrained_request = prefill.constraint.is_some();
        let parser_started = Instant::now();
        let start = self.start_decode(prefill, sender, control).await;
        if constrained_request {
            observe_phase(
                &mut self.metrics.parser_mask_time,
                &mut self.metrics.parser_mask_time_ns,
                parser_started.elapsed(),
            );
        }
        let DecodeStart {
            params,
            prompt_tokens,
            cached_tokens,
            started,
            sampling,
            paged,
            constraint,
            mut guidance,
            mut constrained,
            mut allow,
            mut pipeline,
            mut tool_parser,
        } = match start {
            Ok(start) => start,
            Err(error) => return Err(error),
        };
        let mut consumed = 0_u32;
        let mut unconstrained_consumed = 0_u32;
        let replay_base = self.runner.metrics().submissions;
        let tokenizer = match self.tokenizer() {
            Ok(tokenizer) => tokenizer,
            Err(error) => return Err(error),
        };
        let skip_special_tokens = constraint.as_ref().is_none_or(|plan| plan.forced);
        let mut decoder = tokenizer.decode_stream(skip_special_tokens);
        let mut boundary = BoundaryState::new(&params.stops, constraint.as_ref());
        let mut text = String::new();
        let mut token_ids = Vec::new();
        let generation = async {
            while consumed < params.max_tokens && !pipeline.pending.is_empty() {
                pipeline.activate_next();
                if control.is_cancelled() {
                    return Err(GenerationError::cancelled());
                }
                let has_successor = needs_successor(consumed, params.max_tokens);
                let speculative_constraint = constrained;
                let token = take_pipeline_token(
                    &mut self.runner,
                    &mut pipeline,
                    constrained,
                    speculative_constraint,
                    has_successor,
                    sampling,
                    allow.as_deref(),
                )
                .await?;
                token_ids.push(token);
                if consumed == 0 {
                    let ttft = started.elapsed();
                    observe(&mut self.metrics.ttft, ttft);
                    if cached_tokens == 0 {
                        observe(&mut self.metrics.uncached_ttft, ttft);
                    }
                }
                consumed += 1;
                if !constrained {
                    unconstrained_consumed += 1;
                    synchronize_unconstrained(
                        &mut self.runner,
                        replay_base,
                        unconstrained_consumed,
                    )
                    .await?;
                }
                pipeline.ledger.record(token, has_successor);
                if self.chat_info.stop_token_ids.contains(&token) {
                    return Ok(FinishReason::Stop);
                }
                let piece = decoder.step(token).map_err(|error| {
                    GenerationError::new(format!("cannot decode token: {error}"))
                })?;
                if constrained {
                    let parser_started = Instant::now();
                    if process_constraint_step(
                        &mut self.runner,
                        &mut pipeline,
                        &mut tool_parser,
                        guidance.as_ref(),
                        &mut allow,
                        token,
                        piece,
                        speculative_constraint,
                        sampling,
                        sender,
                        control,
                    )
                    .await?
                    {
                        observe_phase(
                            &mut self.metrics.parser_mask_time,
                            &mut self.metrics.parser_mask_time_ns,
                            parser_started.elapsed(),
                        );
                        return Ok(FinishReason::ToolCalls);
                    }
                    observe_phase(
                        &mut self.metrics.parser_mask_time,
                        &mut self.metrics.parser_mask_time_ns,
                        parser_started.elapsed(),
                    );
                    continue;
                }
                let sse_started = sender.map(|_| Instant::now());
                let action = boundary
                    .process(token, piece, &mut text, sender, control)
                    .await?;
                if let Some(started) = sse_started {
                    observe_phase(
                        &mut self.metrics.sse_time,
                        &mut self.metrics.sse_time_ns,
                        started.elapsed(),
                    );
                }
                if matches!(action, BoundaryAction::Stop) {
                    return Ok(FinishReason::Stop);
                }
                if matches!(action, BoundaryAction::Activate) && consumed < params.max_tokens {
                    let parser_started = Instant::now();
                    let (session, mask) = activate_tool_boundary(
                        &mut pipeline,
                        &mut self.runner,
                        &self.guidance,
                        constraint.as_ref(),
                        sampling,
                        &mut tool_parser,
                        sender,
                        control,
                    )
                    .await?;
                    observe_phase(
                        &mut self.metrics.parser_mask_time,
                        &mut self.metrics.parser_mask_time_ns,
                        parser_started.elapsed(),
                    );
                    guidance = Some(session);
                    allow = Some(mask);
                    constrained = true;
                }
            }
            if !constrained {
                boundary.finish(&mut text, sender, control).await?;
            }
            Ok(FinishReason::Length)
        }
        .await;
        let result = finish_decode_request(
            generation,
            &mut pipeline,
            &mut self.runner,
            tool_parser,
            text,
            token_ids,
            Usage {
                prompt_tokens,
                cached_tokens,
                completion_tokens: consumed,
            },
        )
        .await;
        let ledger = result.is_ok().then(|| std::mem::take(&mut pipeline.ledger));
        let ledger = self.config.serving.prefix_reuse.then_some(ledger).flatten();
        if paged && let Some(ledger) = ledger {
            let publish = self
                .runner
                .paged_prefix()
                .map_err(|error| {
                    GenerationError::unavailable(format!(
                        "paged prefix publication failed: {error:?}"
                    ))
                })?
                .ok_or_else(|| GenerationError::new("paged prefix is unavailable"))
                .and_then(|sequence| {
                    self.prefix_index
                        .as_mut()
                        .ok_or_else(|| GenerationError::new("prefix index is unavailable"))?
                        .publish(self.prefix_namespace, ledger.into_tokens(), &sequence)
                        .map_err(|error| GenerationError::unavailable(error.to_string()))
                });
            publish?;
        }
        result
    }
}

fn needs_successor(consumed: u32, limit: u32) -> bool {
    consumed.checked_add(1).is_some_and(|next| next < limit)
}

async fn finish_decode_request<B>(
    generation: Result<FinishReason, GenerationError>,
    pipeline: &mut DecodePipeline,
    runner: &mut EngineRunner<B>,
    parser: Option<ToolParser>,
    text: String,
    token_ids: Vec<u32>,
    usage: Usage,
) -> Result<Completion, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let cleanup = pipeline.cleanup(runner).await;
    let tool_call = finish_tool_parser(&generation, parser).await;
    finish_completion(generation, cleanup, tool_call, text, token_ids, usage)
}

fn finish_completion(
    generation: Result<FinishReason, GenerationError>,
    cleanup: Result<(), GenerationError>,
    tool_call: Result<Option<GeneratedToolCall>, GenerationError>,
    text: String,
    token_ids: Vec<u32>,
    usage: Usage,
) -> Result<Completion, GenerationError> {
    let finish_reason = match (generation, cleanup) {
        (Ok(reason), Ok(())) => reason,
        (Err(error), _) | (Ok(_), Err(error)) => return Err(error),
    };
    let tool_call = tool_call?;
    Ok(Completion {
        text,
        tool_call,
        finish_reason,
        usage,
        token_ids,
    })
}

type PendingDecode = (EngineDecodeOutput, u32);

struct DecodeStart {
    params: ResolvedParams,
    prompt_tokens: u32,
    cached_tokens: u32,
    started: Instant,
    sampling: SamplingParams,
    paged: bool,
    constraint: Option<ConstraintPlan>,
    guidance: Option<GuidanceSession>,
    constrained: bool,
    allow: Option<Vec<u32>>,
    pipeline: DecodePipeline,
    tool_parser: Option<ToolParser>,
}

async fn prepare_decode<B>(
    runner: &mut EngineRunner<B>,
    worker: &GuidanceWorker,
    prefill: PrefilledPrompt,
    prompt: Arc<StdMutex<PromptArtifacts<ChatFormat>>>,
    sender: Option<&mpsc::Sender<GenerationEvent>>,
    control: &GenerationControl,
) -> Result<DecodeStart, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let PrefilledPrompt {
        mut first,
        params,
        prompt_tokens,
        cached_tokens,
        started,
        sampling,
        paged,
        ledger,
        constraint,
    } = prefill;
    let first_pos = prompt_tokens
        .checked_sub(1)
        .ok_or_else(|| GenerationError::new("prefill position underflowed"))?;
    let constrained = constraint.as_ref().is_some_and(|plan| plan.forced);
    let mut guidance = None;
    let mut allow = None;
    if constrained {
        let (output, session, mask) = initial_constraint(
            runner,
            worker,
            constraint.as_ref(),
            &first,
            first_pos,
            sampling,
        )
        .await?;
        first = output;
        guidance = Some(session);
        allow = Some(mask);
    }
    let tool_parser =
        start_tool_parser(constraint.as_ref(), prompt, constrained, sender, control).await?;
    Ok(DecodeStart {
        params,
        prompt_tokens,
        cached_tokens,
        started,
        sampling,
        paged,
        constraint,
        guidance,
        constrained,
        allow,
        pipeline: DecodePipeline::new(first, first_pos, prompt_tokens, ledger),
        tool_parser,
    })
}

struct DecodePipeline {
    pending: VecDeque<PendingDecode>,
    active: Option<PendingDecode>,
    forward: Option<(EngineTensor, u32)>,
    position: u32,
    ledger: PrefixLedger,
}

impl DecodePipeline {
    fn new(first: EngineDecodeOutput, first_pos: u32, position: u32, ledger: PrefixLedger) -> Self {
        Self {
            pending: VecDeque::from([(first, first_pos)]),
            active: None,
            forward: None,
            position,
            ledger,
        }
    }

    fn activate_next(&mut self) {
        self.active = self.pending.pop_front();
    }

    fn take_active(&mut self) -> Result<PendingDecode, GenerationError> {
        self.active
            .take()
            .ok_or_else(|| GenerationError::new("active decode output is missing"))
    }

    fn push_selected(
        &mut self,
        output: EngineDecodeOutput,
        start_pos: u32,
    ) -> Result<(), GenerationError> {
        self.pending.push_back((output, start_pos));
        self.position = start_pos
            .checked_add(1)
            .ok_or_else(|| GenerationError::new("decode position overflowed"))?;
        Ok(())
    }

    async fn queue_successor<B>(
        &mut self,
        runner: &mut EngineRunner<B>,
        constrained: bool,
        needed: bool,
        sampling: SamplingParams,
    ) -> Result<(), GenerationError>
    where
        B: Backend + Send + Sync + 'static,
    {
        if !needed {
            return Ok(());
        }
        if constrained {
            let logits = begin_forward(runner, self.position).await?;
            self.forward = Some((logits, self.position));
        } else {
            let output = enqueue_unconstrained(runner, self.position, sampling).await?;
            self.push_selected(output, self.position)?;
        }
        Ok(())
    }

    async fn cleanup<B>(&mut self, runner: &mut EngineRunner<B>) -> Result<(), GenerationError>
    where
        B: Backend + Send + Sync + 'static,
    {
        let queued = finalize_sequence(runner, &mut self.pending, self.active.take()).await;
        let forward = self
            .forward
            .take()
            .map_or(Ok(()), |(logits, _)| discard_forward(runner, &logits));
        match (queued, forward) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), _) | (Ok(()), Err(error)) => Err(error),
        }
    }

    async fn discard_pending<B>(
        &mut self,
        runner: &mut EngineRunner<B>,
    ) -> Result<(), GenerationError>
    where
        B: Backend + Send + Sync + 'static,
    {
        finalize_sequence(runner, &mut self.pending, None).await
    }
}

enum BoundaryAction {
    Continue,
    Stop,
    Activate,
}

struct BoundaryState {
    trigger: Option<TokenMatcher>,
    matcher: StopMatcher,
    trigger_stop: Option<usize>,
}

impl BoundaryState {
    fn new(stops: &[String], constraint: Option<&ConstraintPlan>) -> Self {
        let mut stops = stops.to_vec();
        let trigger_stop = constraint.filter(|plan| !plan.forced).map(|plan| {
            let index = stops.len();
            stops.push(plan.trigger.clone());
            index
        });
        let trigger = constraint
            .filter(|plan| !plan.forced)
            .map(|plan| TokenMatcher::new(&plan.trigger_tokens));
        Self {
            trigger,
            matcher: StopMatcher::new(&stops),
            trigger_stop,
        }
    }

    async fn process(
        &mut self,
        token: u32,
        piece: Option<String>,
        text: &mut String,
        sender: Option<&mpsc::Sender<GenerationEvent>>,
        control: &GenerationControl,
    ) -> Result<BoundaryAction, GenerationError> {
        let activated = self
            .trigger
            .as_mut()
            .is_some_and(|matcher| matcher.push(token));
        let Some(piece) = piece else {
            return Ok(BoundaryAction::Continue);
        };
        let output = self
            .matcher
            .push(piece.as_bytes())
            .map_err(|error| GenerationError::new(format!("decoded invalid UTF-8: {error}")))?;
        if output.matched && activated && self.matcher.matched_index() == self.trigger_stop {
            text.push_str(&output.text);
            if let Some(sender) = sender
                && !output.text.is_empty()
            {
                send_event(sender, GenerationEvent::Content(output.text), control).await?;
            }
            Ok(BoundaryAction::Activate)
        } else if emit_output(text, output, sender, control).await? {
            Ok(BoundaryAction::Stop)
        } else {
            Ok(BoundaryAction::Continue)
        }
    }

    async fn finish(
        &mut self,
        text: &mut String,
        sender: Option<&mpsc::Sender<GenerationEvent>>,
        control: &GenerationControl,
    ) -> Result<(), GenerationError> {
        let output = self
            .matcher
            .finish()
            .map_err(|error| GenerationError::new(format!("decoded invalid UTF-8: {error}")))?;
        emit_output(text, output, sender, control).await?;
        Ok(())
    }
}

async fn initial_constraint<B>(
    runner: &mut EngineRunner<B>,
    worker: &GuidanceWorker,
    plan: Option<&ConstraintPlan>,
    first: &EngineDecodeOutput,
    start_pos: u32,
    sampling: SamplingParams,
) -> Result<(EngineDecodeOutput, GuidanceSession, Vec<u32>), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let session = compile_constraint(worker, plan).await?;
    let allow = next_mask(&session).await?;
    let output = finish_selection(runner, &first.logits, start_pos, sampling, &allow).await?;
    Ok((output, session, allow))
}

async fn enqueue_unconstrained<B>(
    runner: &mut EngineRunner<B>,
    start_pos: u32,
    sampling: SamplingParams,
) -> Result<EngineDecodeOutput, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    runner
        .enqueue_decode(EngineDecode {
            phase: forja_host::BatchPhase::Decode,
            tokens: None,
            start_pos,
            sampling,
        })
        .await
        .map_err(|error| GenerationError::unavailable(format!("engine decode failed: {error}")))?
        .map_err(|error| GenerationError::unavailable(format!("engine decode failed: {error:?}")))
}

async fn begin_forward<B>(
    runner: &mut EngineRunner<B>,
    start_pos: u32,
) -> Result<EngineTensor, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    runner
        .begin_decode(EngineBeginDecode {
            phase: forja_host::BatchPhase::Decode,
            tokens: None,
            start_pos,
        })
        .await
        .map_err(|error| {
            GenerationError::unavailable(format!("constrained forward failed: {error}"))
        })?
        .map_err(|error| {
            GenerationError::unavailable(format!("constrained forward failed: {error:?}"))
        })
}

async fn finish_selection<B>(
    runner: &mut EngineRunner<B>,
    logits: &EngineTensor,
    start_pos: u32,
    sampling: SamplingParams,
    allow: &[u32],
) -> Result<EngineDecodeOutput, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    runner
        .finish_decode(logits, start_pos, sampling, allow)
        .await
        .map_err(|error| {
            GenerationError::unavailable(format!("constrained selection failed: {error}"))
        })?
        .map_err(|error| {
            GenerationError::unavailable(format!("constrained selection failed: {error:?}"))
        })
}

async fn selected_token<B>(
    runner: &mut EngineRunner<B>,
    output: EngineDecodeOutput,
) -> Result<u32, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let bytes = runner.read_queued_token(output).await.map_err(|error| {
        GenerationError::unavailable(format!("selected-token read failed: {error:?}"))
    })?;
    read_token(&bytes)
}

async fn take_pipeline_token<B>(
    runner: &mut EngineRunner<B>,
    pipeline: &mut DecodePipeline,
    constrained: bool,
    speculative_constraint: bool,
    has_successor: bool,
    sampling: SamplingParams,
    allow: Option<&[u32]>,
) -> Result<u32, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    pipeline
        .queue_successor(
            runner,
            constrained && !speculative_constraint,
            has_successor,
            sampling,
        )
        .await?;
    let (output, start_pos) = pipeline.take_active()?;
    if speculative_constraint {
        select_speculative_token(
            runner,
            pipeline,
            output,
            start_pos,
            sampling,
            allow,
            has_successor,
        )
        .await
    } else {
        selected_token(runner, output).await
    }
}

#[allow(clippy::too_many_arguments)]
async fn process_constraint_step<B>(
    runner: &mut EngineRunner<B>,
    pipeline: &mut DecodePipeline,
    parser: &mut Option<ToolParser>,
    guidance: Option<&GuidanceSession>,
    allow: &mut Option<Vec<u32>>,
    token: u32,
    piece: Option<String>,
    speculative: bool,
    sampling: SamplingParams,
    sender: Option<&mpsc::Sender<GenerationEvent>>,
    control: &GenerationControl,
) -> Result<bool, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    parse_tool_piece(parser, piece, sender, control).await?;
    if speculative {
        let (complete, next) = advance_speculative_constraint(guidance, token).await?;
        *allow = next;
        Ok(complete)
    } else {
        process_constrained_token(runner, guidance, token, pipeline, sampling).await
    }
}

/// Keeps an allowed ordinary draw and independently redraws rejected tokens under the mask.
///
/// This is rejection sampling: the accepted marginal is the ordinary distribution conditioned
/// on the allowed set. Domain-separating the correction seed preserves deterministic replay while
/// ensuring the rejected draw's uniform is never reused.
async fn select_speculative_token<B>(
    runner: &mut EngineRunner<B>,
    pipeline: &mut DecodePipeline,
    output: EngineDecodeOutput,
    start_pos: u32,
    sampling: SamplingParams,
    allow: Option<&[u32]>,
    has_successor: bool,
) -> Result<u32, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let bytes = runner
        .read_queued_token_retained(&output)
        .await
        .map_err(|error| {
            GenerationError::unavailable(format!("selected-token read failed: {error:?}"))
        })?;
    let provisional = read_token(&bytes)?;
    let allow = allow.ok_or_else(|| GenerationError::new("constraint mask is missing"))?;
    if token_allowed(allow, provisional) {
        runner.release_queued_decode(output).map_err(|error| {
            GenerationError::unavailable(format!("decode output release failed: {error:?}"))
        })?;
        return Ok(provisional);
    }

    pipeline.discard_pending(runner).await?;
    let correction = SamplingParams {
        seed: correction_sampling_seed(sampling.seed),
        ..sampling
    };
    let corrected = finish_selection(runner, &output.logits, start_pos, correction, allow).await?;
    let token = selected_token(runner, corrected).await?;
    let next_pos = start_pos
        .checked_add(1)
        .ok_or_else(|| GenerationError::new("decode position overflowed"))?;
    pipeline.position = next_pos;
    if has_successor {
        let next = enqueue_unconstrained(runner, next_pos, sampling).await?;
        pipeline.push_selected(next, next_pos)?;
    }
    Ok(token)
}

fn token_allowed(allow: &[u32], token: u32) -> bool {
    usize::try_from(token / 32)
        .ok()
        .and_then(|word| allow.get(word))
        .is_some_and(|bits| bits & (1_u32 << (token % 32)) != 0)
}

async fn advance_speculative_constraint(
    session: Option<&GuidanceSession>,
    token: u32,
) -> Result<(bool, Option<Vec<u32>>), GenerationError> {
    let session = session.ok_or_else(|| GenerationError::new("grammar session is missing"))?;
    session
        .commit(token)
        .await
        .map_err(|error| constraint_error(&error))?;
    let next = session
        .mask()
        .await
        .map_err(|error| constraint_error(&error))?;
    Ok((next.is_none(), next))
}

async fn synchronize_unconstrained<B>(
    runner: &mut EngineRunner<B>,
    replay_base: u64,
    consumed: u32,
) -> Result<(), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    if consumed <= 1 {
        return Ok(());
    }
    let expected = replay_base
        .checked_add(u64::from(consumed - 1))
        .ok_or_else(|| GenerationError::new("submission count overflowed"))?;
    runner
        .wait_for_submissions(expected)
        .await
        .map_err(|error| {
            GenerationError::unavailable(format!("engine synchronization failed: {error}"))
        })
}

async fn process_constrained_token<B>(
    runner: &mut EngineRunner<B>,
    session: Option<&GuidanceSession>,
    token: u32,
    pipeline: &mut DecodePipeline,
    sampling: SamplingParams,
) -> Result<bool, GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let session = session.ok_or_else(|| GenerationError::new("grammar session is missing"))?;
    let (complete, next) =
        advance_constraint(runner, session, token, pipeline.forward.take(), sampling).await?;
    if let Some((output, start_pos)) = next {
        pipeline.push_selected(output, start_pos)?;
    }
    Ok(complete)
}

async fn advance_constraint<B>(
    runner: &mut EngineRunner<B>,
    session: &GuidanceSession,
    token: u32,
    forward: Option<(EngineTensor, u32)>,
    sampling: SamplingParams,
) -> Result<(bool, Option<(EngineDecodeOutput, u32)>), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    session
        .commit(token)
        .await
        .map_err(|error| constraint_error(&error))?;
    let allow = session
        .mask()
        .await
        .map_err(|error| constraint_error(&error))?;
    let Some(allow) = allow else {
        if let Some((logits, _)) = forward {
            discard_forward(runner, &logits)?;
        }
        return Ok((true, None));
    };
    let Some((logits, start_pos)) = forward else {
        return Ok((false, None));
    };
    let output = finish_selection(runner, &logits, start_pos, sampling, &allow).await?;
    Ok((false, Some((output, start_pos))))
}

async fn activate_at_boundary<B>(
    pipeline: &mut DecodePipeline,
    runner: &mut EngineRunner<B>,
    worker: &GuidanceWorker,
    plan: Option<&ConstraintPlan>,
    sampling: SamplingParams,
) -> Result<(GuidanceSession, Vec<u32>), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let speculative = pipeline
        .pending
        .pop_front()
        .ok_or_else(|| GenerationError::new("trigger activation has no speculative token"))?;
    let (session, output, rewind_pos, allow) =
        activate_constraint(runner, worker, plan, speculative, sampling).await?;
    pipeline.push_selected(output, rewind_pos)?;
    Ok((session, allow))
}

#[allow(clippy::too_many_arguments)]
async fn activate_tool_boundary<B>(
    pipeline: &mut DecodePipeline,
    runner: &mut EngineRunner<B>,
    worker: &GuidanceWorker,
    plan: Option<&ConstraintPlan>,
    sampling: SamplingParams,
    parser: &mut Option<ToolParser>,
    sender: Option<&mpsc::Sender<GenerationEvent>>,
    control: &GenerationControl,
) -> Result<(GuidanceSession, Vec<u32>), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    seed_tool_parser(parser, sender, control).await?;
    activate_at_boundary(pipeline, runner, worker, plan, sampling).await
}

async fn activate_constraint<B>(
    runner: &mut EngineRunner<B>,
    worker: &GuidanceWorker,
    plan: Option<&ConstraintPlan>,
    speculative: (EngineDecodeOutput, u32),
    sampling: SamplingParams,
) -> Result<(GuidanceSession, EngineDecodeOutput, u32, Vec<u32>), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let (speculative, rewind_pos) = speculative;
    runner
        .discard_queued_decode(speculative)
        .await
        .map_err(|error| {
            GenerationError::unavailable(format!("speculative decode discard failed: {error:?}"))
        })?;
    runner
        .rewind_decode(rewind_pos)
        .await
        .map_err(|error| GenerationError::unavailable(format!("trigger rewind failed: {error}")))?
        .map_err(|error| {
            GenerationError::unavailable(format!("trigger rewind failed: {error:?}"))
        })?;
    let session = compile_constraint(worker, plan).await?;
    let allow = next_mask(&session).await?;
    let logits = begin_forward(runner, rewind_pos).await?;
    let output = finish_selection(runner, &logits, rewind_pos, sampling, &allow).await?;
    Ok((session, output, rewind_pos, allow))
}

fn discard_forward<B>(
    runner: &mut EngineRunner<B>,
    logits: &EngineTensor,
) -> Result<(), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    runner.discard_tensor(logits).map_err(|error| {
        GenerationError::unavailable(format!("constrained forward cleanup failed: {error:?}"))
    })
}

async fn compile_constraint(
    worker: &GuidanceWorker,
    plan: Option<&ConstraintPlan>,
) -> Result<GuidanceSession, GenerationError> {
    let grammar = plan
        .ok_or_else(|| GenerationError::new("constraint plan is missing"))?
        .grammar
        .clone();
    worker
        .compile(grammar)
        .await
        .map_err(|error| constraint_error(&error))
}

async fn next_mask(session: &GuidanceSession) -> Result<Vec<u32>, GenerationError> {
    session
        .mask()
        .await
        .map_err(|error| constraint_error(&error))?
        .ok_or_else(|| GenerationError::new("grammar completed before token selection"))
}

fn constraint_error(error: &str) -> GenerationError {
    GenerationError::invalid(format!("tool grammar is not usable: {error}"))
}

struct TokenMatcher {
    pattern: Vec<u32>,
    failure: Vec<usize>,
    matched: usize,
}

impl TokenMatcher {
    fn new(pattern: &[u32]) -> Self {
        let mut failure = vec![0; pattern.len()];
        for index in 1..pattern.len() {
            let mut prefix = failure[index - 1];
            while prefix > 0 && pattern[index] != pattern[prefix] {
                prefix = failure[prefix - 1];
            }
            if pattern[index] == pattern[prefix] {
                prefix += 1;
            }
            failure[index] = prefix;
        }
        Self {
            pattern: pattern.to_vec(),
            failure,
            matched: 0,
        }
    }

    fn push(&mut self, token: u32) -> bool {
        while self.matched > 0 && token != self.pattern[self.matched] {
            self.matched = self.failure[self.matched - 1];
        }
        if token == self.pattern[self.matched] {
            self.matched += 1;
        }
        if self.matched == self.pattern.len() {
            self.matched = self.failure[self.matched - 1];
            true
        } else {
            false
        }
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
    pending: &mut VecDeque<(EngineDecodeOutput, u32)>,
    active: Option<(EngineDecodeOutput, u32)>,
) -> Result<(), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    let mut failure = None;
    for (output, _) in active.into_iter().chain(pending.drain(..)) {
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
) -> Result<(Vec<u32>, u32, ResolvedParams, Option<ConstraintPlan>), GenerationError>
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
    let constraint_mode = constraint_mode(request);
    let tools = if constraint_mode.is_some() {
        serde_json::to_string(&request.params.tools)
            .map_err(|error| GenerationError::new(format!("cannot encode tools: {error}")))?
    } else {
        "null".to_owned()
    };
    let choice = serde_json::to_string(&request.params.tool_choice)
        .map_err(|error| GenerationError::new(format!("cannot encode tool choice: {error}")))?;
    let constraint_tools = request.params.tools.clone().unwrap_or_default();
    let selected = request
        .params
        .tool_choice
        .as_ref()
        .and_then(|choice| match choice {
            ToolChoice::Named(choice) => Some(choice.function.name.clone()),
            ToolChoice::Mode(_) => None,
        });
    let response = serde_json::to_string(&request.params.response_format)
        .map_err(|error| GenerationError::new(format!("cannot encode response format: {error}")))?;
    let options = serde_json::to_string(&serde_json::json!({
        "add_generation_prompt": true,
        "tool_choice": request.params.tool_choice.as_ref(),
        "parallel_tool_calls": request.params.parallel_tool_calls,
        "response_format": request.params.response_format.as_ref(),
    }))
    .map_err(|error| GenerationError::new(format!("cannot encode render options: {error}")))?;
    let max_rendered_bytes = config.request.max_text_bytes.get();
    let (tokens, constraint) = tokio::task::spawn_blocking(move || {
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
        let constraint = constraint_mode
            .map(|forced| {
                let output = prompt.renderer.grammar(&tools, &choice, &response)?;
                let encoding = prompt
                    .tokenizer
                    .encode(output.trigger.as_str(), false)
                    .map_err(|error| {
                        GenerationError::new(format!("cannot tokenize tool trigger: {error}"))
                    })?;
                let trigger_tokens = encoding.get_ids().to_vec();
                if trigger_tokens.is_empty()
                    || !prompt
                        .tokenizer
                        .decode(&trigger_tokens, false)
                        .is_ok_and(|decoded| decoded == output.trigger)
                {
                    return Err(GenerationError::new(
                        "chat format trigger is not a canonical token sequence",
                    ));
                }
                Ok(ConstraintPlan {
                    grammar: output.grammar,
                    trigger: output.trigger,
                    trigger_tokens,
                    forced,
                    tools: constraint_tools.clone(),
                    selected: selected.clone(),
                })
            })
            .transpose()?;
        let tokens = prompt
            .tokenizer
            .encode(prompt_text, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| GenerationError::new(format!("cannot tokenize prompt: {error}")))?;
        Ok((tokens, constraint))
    })
    .await
    .map_err(|error| GenerationError::new(format!("prompt worker failed: {error}")))??;
    let limits = (max_context, config.request.max_prompt_tokens);
    let finished = finish_prompt(tokens, params, limits, control, "rendered prompt")?;
    let (tokens, prompt_tokens, params) = finished;
    Ok((tokens, prompt_tokens, params, constraint))
}

fn constraint_mode(request: &ChatCompletionRequest) -> Option<bool> {
    request
        .params
        .tools
        .as_ref()
        .filter(|tools| !tools.is_empty())?;
    match request.params.tool_choice.as_ref() {
        Some(ToolChoice::Mode(mode)) if mode == "none" => None,
        Some(ToolChoice::Mode(mode)) => Some(mode == "required"),
        Some(ToolChoice::Named(_)) => Some(true),
        None => Some(false),
    }
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
    let limits = (max_context, config.request.max_prompt_tokens);
    finish_prompt(tokens, params, limits, control, "prompt")
}

fn finish_prompt(
    tokens: Vec<u32>,
    params: ResolvedParams,
    limits: (u32, Option<u32>),
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
    let params = validate_prompt(params, prompt_tokens, limits.0, limits.1).map_err(|error| {
        if error.is_context_length() {
            GenerationError::context_length(error.to_string())
        } else {
            GenerationError::invalid(error.to_string())
        }
    })?;
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

    fn serving_info() -> forja_host::ServingInfo {
        forja_host::ServingInfo {
            page_tokens: 64,
            planes: Vec::new(),
            active_lanes: 8,
            max_context: 4_096,
            prefill_chunk_multiple: None,
            batch_shapes: [1, 2, 4, 8]
                .into_iter()
                .map(|bucket| forja_host::BatchShape {
                    phase: forja_host::BatchPhase::Decode,
                    bucket,
                    tokens: 1,
                    selection: forja_host::BatchSelection::Combined,
                })
                .collect(),
        }
    }

    #[test]
    fn scheduler_caps_active_lanes_to_declared_decode_buckets() {
        assert_eq!(scheduler_decode_buckets(Vec::new(), 1), (1, Vec::new()));
        assert_eq!(scheduler_decode_buckets(vec![1], 8), (1, vec![1]));
        assert_eq!(
            scheduler_decode_buckets(vec![1, 2, 4, 8], 6),
            (6, vec![1, 2, 4, 8])
        );
        assert_eq!(
            scheduler_decode_buckets(vec![1, 2, 4, 8], 3),
            (3, vec![1, 2, 4])
        );
    }

    #[test]
    fn explicit_scheduler_limits_must_lower_engine_limits() {
        let info = EngineInfo {
            prompt_block_tokens: None,
            vocab: 1,
            max_context: 4_096,
            tap_layers: Vec::new(),
            router_layers: Vec::new(),
            serving: Some(serving_info()),
        };
        let mut config = HostConfig::default();
        config.serving.max_active_sequences = Choice::Fixed(std::num::NonZeroU32::new(4).unwrap());
        config.serving.max_batch_size = Choice::Fixed(std::num::NonZeroU32::new(2).unwrap());
        config.serving.max_prefill_chunks = Choice::Fixed(std::num::NonZeroU32::new(1).unwrap());
        config.serving.prompt_token_budget = Choice::Fixed(std::num::NonZeroU64::new(64).unwrap());
        config.serving.idle_prompt_token_budget =
            Choice::Fixed(std::num::NonZeroU64::new(1024).unwrap());
        let resolved = SchedulerConfig::resolve(&info, &config, 128, 2048).unwrap();
        assert_eq!(resolved.active_lanes, 2);
        assert_eq!(resolved.max_prefill_chunks, 1);
        assert_eq!(resolved.prompt_token_budget, 64);
        assert_eq!(resolved.idle_prompt_token_budget, 1024);
        let mut smaller = info.clone();
        smaller
            .serving
            .as_mut()
            .unwrap()
            .batch_shapes
            .push(forja_host::BatchShape {
                phase: BatchPhase::Prompt,
                bucket: 1,
                tokens: 32,
                selection: BatchSelection::Combined,
            });
        let capped = SchedulerConfig::resolve(&smaller, &config, 128, 2048).unwrap();
        assert_eq!(
            (capped.prompt_token_budget, capped.idle_prompt_token_budget),
            (32, 32)
        );

        config.serving.idle_prompt_token_budget =
            Choice::Fixed(std::num::NonZeroU64::new(2049).unwrap());
        assert!(
            SchedulerConfig::resolve(&info, &config, 128, 2048)
                .unwrap_err()
                .to_string()
                .contains("serving.idle_prompt_token_budget")
        );
        config.serving.max_active_sequences = Choice::Fixed(std::num::NonZeroU32::new(9).unwrap());
        assert!(
            SchedulerConfig::resolve(&info, &config, 128, 2048)
                .unwrap_err()
                .to_string()
                .contains("serving.max_active_sequences")
        );
    }

    #[test]
    fn atomic_prompt_blocks_refuse_lower_budgets_without_paged_serving() {
        let info = EngineInfo {
            prompt_block_tokens: std::num::NonZeroU32::new(64),
            vocab: 1,
            max_context: 4096,
            tap_layers: Vec::new(),
            router_layers: Vec::new(),
            serving: None,
        };
        let mut config = HostConfig::default();
        let resolved = SchedulerConfig::resolve(&info, &config, 64, 64).unwrap();
        assert_eq!(resolved.prompt_token_budget, 64);
        assert_eq!(resolved.prefill_chunk_multiple, info.prompt_block_tokens);
        config.serving.prompt_token_budget = Choice::Fixed(std::num::NonZeroU64::new(32).unwrap());
        assert!(SchedulerConfig::resolve(&info, &config, 64, 64).is_err());
        config.serving.prompt_token_budget = Choice::Auto;
        config.serving.idle_prompt_token_budget =
            Choice::Fixed(std::num::NonZeroU64::new(32).unwrap());
        assert!(SchedulerConfig::resolve(&info, &config, 64, 64).is_err());
        config.serving.idle_prompt_token_budget = Choice::Auto;
        assert!(SchedulerConfig::resolve(&info, &config, 32, 64).is_err());
        assert!(SchedulerConfig::resolve(&info, &config, 64, 33).is_err());
    }

    #[test]
    fn atomic_prompt_blocks_validate_final_serving_geometry() {
        let mut info = EngineInfo {
            prompt_block_tokens: std::num::NonZeroU32::new(64),
            vocab: 1,
            max_context: 4096,
            tap_layers: vec![],
            router_layers: vec![],
            serving: Some(serving_info()),
        };
        let config = HostConfig::default();
        for (tokens, multiple, valid) in [
            (32, 64, false),
            (64, 32, false),
            (128, 96, false),
            (64, 64, true),
            (128, 128, true),
        ] {
            let serving = info.serving.as_mut().unwrap();
            serving.prefill_chunk_multiple = std::num::NonZeroU32::new(multiple);
            serving
                .batch_shapes
                .retain(|shape| shape.phase != BatchPhase::Prompt);
            serving.batch_shapes.push(forja_host::BatchShape {
                phase: BatchPhase::Prompt,
                bucket: 1,
                tokens,
                selection: BatchSelection::Combined,
            });
            assert_eq!(
                SchedulerConfig::resolve(&info, &config, 128, 128).is_ok(),
                valid,
                "tokens={tokens}, multiple={multiple}"
            );
        }
        info.serving.as_mut().unwrap().prefill_chunk_multiple = None;
        let resolved = SchedulerConfig::resolve(&info, &config, 128, 128).unwrap();
        assert_eq!(resolved.prefill_chunk_multiple, info.prompt_block_tokens);
    }

    #[test]
    fn operator_limits_can_disable_chunk_merging() {
        let mut serving = serving_info();
        serving.prefill_chunk_multiple = std::num::NonZeroU32::new(512);
        let info = EngineInfo {
            prompt_block_tokens: None,
            vocab: 1,
            max_context: 4096,
            tap_layers: vec![],
            router_layers: vec![],
            serving: Some(serving),
        };
        let mut config = HostConfig::default();
        let resolved = SchedulerConfig::resolve(&info, &config, 512, 2048).unwrap();
        assert_eq!(
            resolved
                .prefill_chunk_multiple
                .map(std::num::NonZeroU32::get),
            Some(512)
        );
        config.serving.prompt_token_budget = Choice::Fixed(std::num::NonZeroU64::new(128).unwrap());
        assert!(
            SchedulerConfig::resolve(&info, &config, 512, 2048)
                .unwrap()
                .prefill_chunk_multiple
                .is_none()
        );
    }

    #[test]
    fn explicit_prefix_limits_bound_the_derived_index() {
        let mut config = HostConfig::default();
        config.server.queue_length = 16;
        assert_eq!(
            prefix_limits(&serving_info(), &config, 2_048).unwrap(),
            PrefixLimits {
                entries: 16,
                tokens: 32_768,
                referenced_pages: 512,
            }
        );
        config.serving.prefix_cache_entries = Choice::Fixed(std::num::NonZeroU32::new(3).unwrap());
        config.serving.prefix_cache_tokens =
            Choice::Fixed(std::num::NonZeroU64::new(1_000).unwrap());
        config.serving.prefix_cache_pages = Choice::Fixed(std::num::NonZeroU32::new(20).unwrap());
        assert_eq!(
            prefix_limits(&serving_info(), &config, 2_048).unwrap(),
            PrefixLimits {
                entries: 3,
                tokens: 1_000,
                referenced_pages: 20,
            }
        );

        config.serving.prefix_cache_entries = Choice::Fixed(std::num::NonZeroU32::new(17).unwrap());
        assert!(
            prefix_limits(&serving_info(), &config, 2_048)
                .unwrap_err()
                .to_string()
                .contains("serving.prefix_cache_entries")
        );
    }

    #[test]
    fn preserves_in_flight_paged_bank_classification() {
        let source = forja_host::bindings::l9o::gpu::compute::Error::Quota(
            forja_core::PagePoolError::BankInFlight.to_string(),
        );
        let error = GenerationError::paged_admission(&source);
        assert!(error.paged_bank_in_flight());
    }
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

    #[test]
    fn token_trigger_matcher_preserves_overlapping_prefixes() {
        let mut matcher = TokenMatcher::new(&[1, 2, 1]);
        let hits = [1, 2, 1, 2, 1].map(|token| matcher.push(token));
        assert_eq!(hits, [false, false, true, false, true]);
    }

    #[test]
    fn constraint_mode_distinguishes_lazy_and_forced_choices() {
        let request = |choice| {
            serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
                "model": "test",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function", "function": {
                    "name": "read", "parameters": {"type": "object"}
                }}],
                "tool_choice": choice,
            }))
            .unwrap()
        };
        assert_eq!(
            constraint_mode(&request(serde_json::json!("auto"))),
            Some(false)
        );
        assert_eq!(
            constraint_mode(&request(serde_json::json!("required"))),
            Some(true)
        );
        assert_eq!(constraint_mode(&request(serde_json::json!("none"))), None);
        assert_eq!(
            constraint_mode(&request(serde_json::json!({
                "type": "function", "function": {"name": "read"}
            }))),
            Some(true)
        );
    }

    #[tokio::test]
    async fn lazy_constraint_preserves_output_before_its_trigger() {
        let plan = ConstraintPlan {
            grammar: String::new(),
            trigger: "<tool_call>\n".to_owned(),
            trigger_tokens: vec![7, 8],
            forced: false,
            tools: Vec::new(),
            selected: None,
        };
        let control = GenerationControl::new(Duration::from_secs(1));
        let mut ordinary = BoundaryState::new(&[], None);
        let mut lazy = BoundaryState::new(&[], Some(&plan));
        let mut ordinary_text = String::new();
        let mut lazy_text = String::new();
        for (token, piece) in [(1, "ordinary "), (2, "answer")] {
            let ordinary_action = ordinary
                .process(
                    token,
                    Some(piece.to_owned()),
                    &mut ordinary_text,
                    None,
                    &control,
                )
                .await
                .unwrap();
            let lazy_action = lazy
                .process(
                    token,
                    Some(piece.to_owned()),
                    &mut lazy_text,
                    None,
                    &control,
                )
                .await
                .unwrap();
            assert!(matches!(ordinary_action, BoundaryAction::Continue));
            assert!(matches!(lazy_action, BoundaryAction::Continue));
        }
        ordinary
            .finish(&mut ordinary_text, None, &control)
            .await
            .unwrap();
        lazy.finish(&mut lazy_text, None, &control).await.unwrap();
        assert_eq!(lazy_text, ordinary_text);
    }

    #[tokio::test]
    async fn component_events_build_one_validated_call() {
        let tool = serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": null,
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": false
                }
            }
        }))
        .unwrap();
        let plan = ConstraintPlan {
            grammar: String::new(),
            trigger: "<tool_call>\n".to_owned(),
            trigger_tokens: vec![1],
            forced: true,
            tools: vec![tool],
            selected: None,
        };
        let format = ChatFormat::load(
            test_guests::qwen3_coder_chat_format(),
            crate::ChatFormatLimits::default(),
        )
        .unwrap();
        let prompt = Arc::new(StdMutex::new(PromptArtifacts {
            tokenizer: Tokenizer::new(tokenizers::models::bpe::BPE::default()),
            renderer: format,
        }));
        let mut parser = ToolParser::new(&plan, prompt).unwrap();
        assert_eq!(parser.seed().await.unwrap().len(), 0);
        parser
            .parse(
                "<function=read_file>\n<parameter=path>\na.rs\n</parameter>\n</function>\n</tool_call>"
                    .to_owned(),
            )
            .await
            .unwrap();
        assert_eq!(parser.finish().unwrap().arguments, r#"{"path":"a.rs"}"#);
        parser.reset().await;
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
