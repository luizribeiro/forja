use std::{
    collections::VecDeque,
    sync::{Arc, Mutex as StdMutex},
    time::Instant,
};

use forja_core::Backend;
use tokio::sync::{mpsc, oneshot};

use super::{
    ConstraintPlan, ServingEngine, prepare_chat_prompt, prepare_completion_prompt, send_terminal,
};
use crate::{
    ChatCompletionRequest, Completion, CompletionParams, CompletionRequest, GenerationControl,
    GenerationError, GenerationEvent, ResolvedParams, ServingMetrics, http::ServiceCommand,
    metrics::observe_phase,
};

mod batch;
mod lane;
mod prefill;

enum CompletionSink {
    Complete(oneshot::Sender<Result<Completion, GenerationError>>),
    Stream {
        events: mpsc::Sender<GenerationEvent>,
        reply: oneshot::Sender<Result<(), GenerationError>>,
    },
}

impl CompletionSink {
    async fn finish(
        self,
        params: &CompletionParams,
        completion: Completion,
        control: &GenerationControl,
    ) {
        match self {
            Self::Complete(reply) => drop(reply.send(Ok(completion))),
            Self::Stream { events, reply } => {
                drop(reply.send(send_terminal(&events, params, completion, control).await));
            }
        }
    }

    fn fail(self, error: GenerationError) {
        match self {
            Self::Complete(reply) => drop(reply.send(Err(error))),
            Self::Stream { reply, .. } => drop(reply.send(Err(error))),
        }
    }

    fn events(&self) -> Option<&mpsc::Sender<GenerationEvent>> {
        match self {
            Self::Complete(_) => None,
            Self::Stream { events, .. } => Some(events),
        }
    }
}

struct PreparedRequest {
    tokens: Vec<u32>,
    prompt_tokens: u32,
    params: ResolvedParams,
    request_params: CompletionParams,
    constraint: Option<ConstraintPlan>,
    control: GenerationControl,
    sink: CompletionSink,
    started: Instant,
}

impl PreparedRequest {
    #[allow(dead_code)]
    fn reserved_tokens(&self) -> Option<u64> {
        u64::from(self.prompt_tokens).checked_add(u64::from(self.params.max_tokens))
    }

    async fn run_single<B>(self, engine: &mut ServingEngine<B>)
    where
        B: Backend + Send + Sync + 'static,
    {
        let Self {
            tokens,
            prompt_tokens,
            params,
            request_params,
            constraint,
            control,
            sink,
            started: _,
        } = self;
        let result = match engine
            .prefill(tokens, prompt_tokens, params, constraint, &control)
            .await
        {
            Ok(prefill) => engine.decode(prefill, sink.events(), &control).await,
            Err(error) => Err(error),
        };
        match result {
            Ok(completion) => sink.finish(&request_params, completion, &control).await,
            Err(error) => sink.fail(error),
        }
    }
}

enum RequestEnvelope {
    Chat {
        request: ChatCompletionRequest,
        control: GenerationControl,
        reply: oneshot::Sender<Result<Completion, GenerationError>>,
        queued: Instant,
    },
    Text {
        request: CompletionRequest,
        control: GenerationControl,
        reply: oneshot::Sender<Result<Completion, GenerationError>>,
        queued: Instant,
    },
    StreamChat {
        request: ChatCompletionRequest,
        events: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
        reply: oneshot::Sender<Result<(), GenerationError>>,
        queued: Instant,
    },
    StreamText {
        request: CompletionRequest,
        events: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
        reply: oneshot::Sender<Result<(), GenerationError>>,
        queued: Instant,
    },
}

impl RequestEnvelope {
    fn from_command(command: ServiceCommand) -> Self {
        match command {
            ServiceCommand::Chat {
                request,
                control,
                reply,
                queued,
            } => Self::Chat {
                request,
                control,
                reply,
                queued,
            },
            ServiceCommand::Text {
                request,
                control,
                reply,
                queued,
            } => Self::Text {
                request,
                control,
                reply,
                queued,
            },
            ServiceCommand::StreamChat {
                request,
                events,
                control,
                reply,
                queued,
            } => Self::StreamChat {
                request,
                events,
                control,
                reply,
                queued,
            },
            ServiceCommand::StreamText {
                request,
                events,
                control,
                reply,
                queued,
            } => Self::StreamText {
                request,
                events,
                control,
                reply,
                queued,
            },
        }
    }

    fn queued(&self) -> Instant {
        match self {
            Self::Chat { queued, .. }
            | Self::Text { queued, .. }
            | Self::StreamChat { queued, .. }
            | Self::StreamText { queued, .. } => *queued,
        }
    }

    fn requires_serial_path(&self) -> bool {
        matches!(
            self,
            Self::Chat { request, .. } | Self::StreamChat { request, .. }
                if request.params.tools.is_some()
        )
    }

    async fn prepare<B>(self, engine: &mut ServingEngine<B>) -> Option<PreparedRequest>
    where
        B: Backend + Send + Sync + 'static,
    {
        let started = Instant::now();
        let (prepared, request_params, control, sink) = match self {
            Self::Chat {
                request,
                control,
                reply,
                queued: _,
            } => {
                let params = request.params.clone();
                let prepared = prepare_chat(engine, &request, &control).await;
                (prepared, params, control, CompletionSink::Complete(reply))
            }
            Self::Text {
                request,
                control,
                reply,
                queued: _,
            } => {
                let params = request.params.clone();
                let prepared = prepare_text(engine, &request, &control).await;
                (prepared, params, control, CompletionSink::Complete(reply))
            }
            Self::StreamChat {
                request,
                events,
                control,
                reply,
                queued: _,
            } => {
                let params = request.params.clone();
                let prepared = prepare_chat(engine, &request, &control).await;
                (
                    prepared,
                    params,
                    control,
                    CompletionSink::Stream { events, reply },
                )
            }
            Self::StreamText {
                request,
                events,
                control,
                reply,
                queued: _,
            } => {
                let params = request.params.clone();
                let prepared = prepare_text(engine, &request, &control).await;
                (
                    prepared,
                    params,
                    control,
                    CompletionSink::Stream { events, reply },
                )
            }
        };
        match prepared {
            Ok((tokens, prompt_tokens, params, constraint)) => Some(PreparedRequest {
                tokens,
                prompt_tokens,
                params,
                request_params,
                constraint,
                control,
                sink,
                started,
            }),
            Err(error) => {
                sink.fail(error);
                None
            }
        }
    }
}

async fn prepare_chat<B>(
    engine: &mut ServingEngine<B>,
    request: &ChatCompletionRequest,
    control: &GenerationControl,
) -> Result<(Vec<u32>, u32, ResolvedParams, Option<ConstraintPlan>), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    prepare_chat_prompt(
        request,
        &engine.config,
        &engine.chat_info,
        engine.max_context,
        engine.info.vocab,
        Arc::clone(&engine.prompt),
        control,
    )
    .await
}

async fn prepare_text<B>(
    engine: &mut ServingEngine<B>,
    request: &CompletionRequest,
    control: &GenerationControl,
) -> Result<(Vec<u32>, u32, ResolvedParams, Option<ConstraintPlan>), GenerationError>
where
    B: Backend + Send + Sync + 'static,
{
    prepare_completion_prompt(
        request,
        &engine.config,
        &engine.chat_info,
        engine.max_context,
        engine.info.vocab,
        Arc::clone(&engine.prompt),
        control,
    )
    .await
    .map(|(tokens, count, resolved)| (tokens, count, resolved, None))
}

impl<B> ServingEngine<B>
where
    B: Backend + Send + Sync + 'static,
{
    pub(crate) async fn run_continuous(
        mut self,
        mut receiver: mpsc::Receiver<ServiceCommand>,
        shared_metrics: Arc<StdMutex<ServingMetrics>>,
    ) {
        let mut pending = VecDeque::new();
        loop {
            if pending.is_empty() {
                let Some(command) = receiver.recv().await else {
                    return;
                };
                pending.push_back(RequestEnvelope::from_command(command));
            }
            while let Ok(command) = receiver.try_recv() {
                pending.push_back(RequestEnvelope::from_command(command));
            }
            let scheduled = Instant::now();
            if self.info.serving.is_some()
                && pending
                    .front()
                    .is_some_and(|request| !request.requires_serial_path())
            {
                self.run_batch_lanes(&mut receiver, &mut pending).await;
            } else if let Some(envelope) = pending.pop_front() {
                observe_phase(
                    &mut self.metrics.queue_time,
                    &mut self.metrics.queue_time_ns,
                    envelope.queued().elapsed(),
                );
                if let Some(prepared) = envelope.prepare(&mut self).await {
                    prepared.run_single(&mut self).await;
                }
            }
            if let Ok(mut metrics) = shared_metrics.lock() {
                self.refresh_runtime_metrics();
                observe_phase(
                    &mut self.metrics.scheduler_time,
                    &mut self.metrics.scheduler_time_ns,
                    scheduled.elapsed(),
                );
                *metrics = self.metrics.clone();
            }
        }
    }
}

fn batch_admission_error(
    error: &forja_host::bindings::l9o::gpu::compute::Error,
) -> GenerationError {
    GenerationError::unavailable(format!("paged admission failed: {error:?}"))
}

fn read_batch_tokens(bytes: &[u8]) -> Result<Vec<u32>, GenerationError> {
    let (tokens, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(GenerationError::new(
            "batch token output is not u32-aligned",
        ));
    }
    Ok(tokens.iter().copied().map(u32::from_le_bytes).collect())
}

fn repeated_error(error: &GenerationError) -> GenerationError {
    GenerationError::unavailable(error.to_string())
}
