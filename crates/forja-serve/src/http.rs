use std::{
    collections::BTreeMap,
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex as StdMutex, RwLock,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, FromRequest, Path, Request, State, rejection::BytesRejection},
    http::{StatusCode, header},
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use forja_core::Backend;
use serde_json::{Value, json};
use tokio::sync::{mpsc, oneshot};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use crate::engine::ErrorKind;
use crate::{
    ChatCompletionRequest, Completion, CompletionRequest, GenerationControl, GenerationError,
    GenerationEvent, ServingEngine, ServingMetrics, Usage, chat_response, text_response,
};

static NEXT_REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Asynchronous chat generation used by the HTTP transport.
pub trait ChatService: Send + 'static {
    /// Served model identifier.
    fn model(&self) -> &str;

    /// Maximum bytes accepted before JSON parsing.
    fn max_body_bytes(&self) -> usize;

    /// Maximum number of requests waiting behind the active sequence.
    fn queue_length(&self) -> u32;

    /// End-to-end request deadline.
    fn request_deadline(&self) -> Duration;

    /// Maximum time a streaming event may remain backpressured.
    fn write_stall_timeout(&self) -> Duration;

    /// Snapshot of currently accumulated bounded metrics.
    fn metrics(&self) -> ServingMetrics {
        ServingMetrics::default()
    }

    /// Runs the bounded command mailbox owned by the HTTP transport.
    #[doc(hidden)]
    fn run(
        self,
        receiver: mpsc::Receiver<ServiceCommand>,
        metrics: Arc<StdMutex<ServingMetrics>>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>>
    where
        Self: Sized,
    {
        Box::pin(run_serial(self, receiver, metrics))
    }

    /// Completes one chat request.
    fn complete(
        &mut self,
        request: ChatCompletionRequest,
        control: GenerationControl,
    ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>;

    /// Completes one unformatted text request.
    fn complete_text(
        &mut self,
        request: CompletionRequest,
        control: GenerationControl,
    ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>;

    /// Streams one chat request.
    fn stream_chat(
        &mut self,
        request: ChatCompletionRequest,
        sender: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
    ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>>;

    /// Streams one unformatted text request.
    fn stream_text(
        &mut self,
        request: CompletionRequest,
        sender: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
    ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>>;
}

impl<B> ChatService for ServingEngine<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn run(
        self,
        receiver: mpsc::Receiver<ServiceCommand>,
        metrics: Arc<StdMutex<ServingMetrics>>,
    ) -> Pin<Box<dyn Future<Output = ()> + Send>> {
        if self.supports_continuous_batching() {
            Box::pin(self.run_continuous(receiver, metrics))
        } else {
            Box::pin(run_serial(self, receiver, metrics))
        }
    }

    fn model(&self) -> &str {
        self.model()
    }

    fn max_body_bytes(&self) -> usize {
        self.max_body_bytes()
    }

    fn queue_length(&self) -> u32 {
        self.queue_length()
    }

    fn request_deadline(&self) -> Duration {
        self.request_deadline()
    }

    fn write_stall_timeout(&self) -> Duration {
        self.write_stall_timeout()
    }

    fn metrics(&self) -> ServingMetrics {
        self.metrics().clone()
    }

    fn complete(
        &mut self,
        request: ChatCompletionRequest,
        control: GenerationControl,
    ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>> {
        Box::pin(async move { self.complete_chat_controlled(&request, &control).await })
    }

    fn complete_text(
        &mut self,
        request: CompletionRequest,
        control: GenerationControl,
    ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>> {
        Box::pin(async move { self.complete_text_controlled(&request, &control).await })
    }

    fn stream_chat(
        &mut self,
        request: ChatCompletionRequest,
        sender: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
    ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
        Box::pin(async move {
            self.stream_chat_controlled(&request, &sender, &control)
                .await
        })
    }

    fn stream_text(
        &mut self,
        request: CompletionRequest,
        sender: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
    ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
        Box::pin(async move {
            self.stream_text_controlled(&request, &sender, &control)
                .await
        })
    }
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActorError {
    Full,
    Closed,
}

#[doc(hidden)]
#[allow(missing_docs)]
pub enum ServiceCommand {
    Chat {
        request: ChatCompletionRequest,
        control: GenerationControl,
        reply: oneshot::Sender<Result<Completion, GenerationError>>,
        queued: std::time::Instant,
    },
    Text {
        request: CompletionRequest,
        control: GenerationControl,
        reply: oneshot::Sender<Result<Completion, GenerationError>>,
        queued: std::time::Instant,
    },
    StreamChat {
        request: ChatCompletionRequest,
        events: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
        reply: oneshot::Sender<Result<(), GenerationError>>,
        queued: std::time::Instant,
    },
    StreamText {
        request: CompletionRequest,
        events: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
        reply: oneshot::Sender<Result<(), GenerationError>>,
        queued: std::time::Instant,
    },
}

impl ServiceCommand {
    pub(crate) fn queued(&self) -> std::time::Instant {
        match self {
            Self::Chat { queued, .. }
            | Self::Text { queued, .. }
            | Self::StreamChat { queued, .. }
            | Self::StreamText { queued, .. } => *queued,
        }
    }
}

struct ServiceActor<S> {
    sender: mpsc::Sender<ServiceCommand>,
    metrics: Arc<StdMutex<ServingMetrics>>,
    marker: std::marker::PhantomData<fn() -> S>,
}

impl<S> Clone for ServiceActor<S> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            metrics: Arc::clone(&self.metrics),
            marker: std::marker::PhantomData,
        }
    }
}

impl<S: ChatService> ServiceActor<S> {
    fn spawn(service: S, capacity: usize) -> Self {
        let (sender, receiver) = mpsc::channel::<ServiceCommand>(capacity.max(1));
        let metrics = Arc::new(StdMutex::new(service.metrics()));
        let actor_metrics = Arc::clone(&metrics);
        tokio::spawn(service.run(receiver, actor_metrics));
        Self {
            sender,
            metrics,
            marker: std::marker::PhantomData,
        }
    }

    fn complete(
        &self,
        request: ChatCompletionRequest,
        control: GenerationControl,
    ) -> Result<oneshot::Receiver<Result<Completion, GenerationError>>, ActorError> {
        let (reply, receiver) = oneshot::channel();
        self.send(ServiceCommand::Chat {
            request,
            control,
            reply,
            queued: std::time::Instant::now(),
        })?;
        Ok(receiver)
    }

    fn complete_text(
        &self,
        request: CompletionRequest,
        control: GenerationControl,
    ) -> Result<oneshot::Receiver<Result<Completion, GenerationError>>, ActorError> {
        let (reply, receiver) = oneshot::channel();
        self.send(ServiceCommand::Text {
            request,
            control,
            reply,
            queued: std::time::Instant::now(),
        })?;
        Ok(receiver)
    }

    fn stream_chat(
        &self,
        request: ChatCompletionRequest,
        events: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
    ) -> Result<oneshot::Receiver<Result<(), GenerationError>>, ActorError> {
        let (reply, receiver) = oneshot::channel();
        self.send(ServiceCommand::StreamChat {
            request,
            events,
            control,
            reply,
            queued: std::time::Instant::now(),
        })?;
        Ok(receiver)
    }

    fn stream_text(
        &self,
        request: CompletionRequest,
        events: mpsc::Sender<GenerationEvent>,
        control: GenerationControl,
    ) -> Result<oneshot::Receiver<Result<(), GenerationError>>, ActorError> {
        let (reply, receiver) = oneshot::channel();
        self.send(ServiceCommand::StreamText {
            request,
            events,
            control,
            reply,
            queued: std::time::Instant::now(),
        })?;
        Ok(receiver)
    }

    fn send(&self, command: ServiceCommand) -> Result<(), ActorError> {
        self.sender.try_send(command).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => ActorError::Full,
            mpsc::error::TrySendError::Closed(_) => ActorError::Closed,
        })
    }

    fn metrics(&self) -> ServingMetrics {
        self.metrics
            .lock()
            .map_or_else(|_| ServingMetrics::default(), |metrics| metrics.clone())
    }
}

async fn run_serial<S: ChatService>(
    mut service: S,
    mut receiver: mpsc::Receiver<ServiceCommand>,
    metrics: Arc<StdMutex<ServingMetrics>>,
) {
    while let Some(command) = receiver.recv().await {
        let queued = command.queued().elapsed();
        let scheduled = std::time::Instant::now();
        match command {
            ServiceCommand::Chat {
                request,
                control,
                reply,
                ..
            } => drop(reply.send(service.complete(request, control).await)),
            ServiceCommand::Text {
                request,
                control,
                reply,
                ..
            } => drop(reply.send(service.complete_text(request, control).await)),
            ServiceCommand::StreamChat {
                request,
                events,
                control,
                reply,
                ..
            } => drop(reply.send(service.stream_chat(request, events, control).await)),
            ServiceCommand::StreamText {
                request,
                events,
                control,
                reply,
                ..
            } => drop(reply.send(service.stream_text(request, events, control).await)),
        }
        if let Ok(mut snapshot) = metrics.lock() {
            *snapshot = service.metrics();
            let snapshot = &mut *snapshot;
            crate::metrics::observe_phase(
                &mut snapshot.queue_time,
                &mut snapshot.queue_time_ns,
                queued,
            );
            crate::metrics::observe_phase(
                &mut snapshot.scheduler_time,
                &mut snapshot.scheduler_time_ns,
                scheduled.elapsed(),
            );
        }
    }
}

/// Builds the OpenAI-compatible chat-completions router.
pub fn chat_router<S: ChatService>(service: S) -> Router {
    serving_router(service).0
}

/// Builds a serving router and a handle that drains its admitted work.
pub fn serving_router<S: ChatService>(service: S) -> (Router, DrainHandle<S>) {
    let max_body_bytes = service.max_body_bytes();
    let generation = PublishedGeneration::new(0, service);
    let state = Arc::new(HttpState {
        generation: RwLock::new(generation),
        ready: std::sync::atomic::AtomicBool::new(true),
        closed: std::sync::atomic::AtomicBool::new(false),
        active: StdMutex::new(BTreeMap::new()),
        drained: tokio::sync::Notify::new(),
    });
    let router = Router::new()
        .route("/v1/chat/completions", post(chat::<S>))
        .route("/v1/completions", post(text::<S>))
        .route("/v1/models", get(models::<S>))
        .route("/v1/models/{model}", get(model::<S>))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(Arc::clone(&state));
    (router, DrainHandle { state })
}

/// Builds the read-only health and Prometheus router for a serving handle.
///
/// # Errors
///
/// Returns an error when the configured metrics path conflicts with a health path.
pub fn operational_router<S: ChatService>(
    handle: &DrainHandle<S>,
    metrics_path: &str,
) -> Result<Router, String> {
    if !metrics_path.starts_with('/') || matches!(metrics_path, "/health/live" | "/health/ready") {
        return Err("metrics path must be an absolute, non-health path".to_owned());
    }
    Ok(Router::new()
        .route("/health/live", get(live))
        .route("/health/ready", get(ready::<S>))
        .route(metrics_path, get(prometheus::<S>))
        .with_state(Arc::clone(&handle.state)))
}

struct HttpState<S> {
    generation: RwLock<PublishedGeneration<S>>,
    ready: std::sync::atomic::AtomicBool,
    closed: std::sync::atomic::AtomicBool,
    active: StdMutex<BTreeMap<u64, ActiveRequest>>,
    drained: tokio::sync::Notify,
}

struct ActiveRequest {
    generation: u64,
    control: GenerationControl,
}

struct PublishedGeneration<S> {
    id: u64,
    model: String,
    deadline: Duration,
    write_stall_timeout: Duration,
    service: ServiceActor<S>,
}

impl<S: ChatService> PublishedGeneration<S> {
    fn new(id: u64, service: S) -> Self {
        let model = service.model().to_owned();
        let deadline = service.request_deadline();
        let write_stall_timeout = service.write_stall_timeout();
        let queue_capacity = usize::try_from(service.queue_length()).unwrap_or(usize::MAX);
        Self {
            id,
            model,
            deadline,
            write_stall_timeout,
            service: ServiceActor::spawn(service, queue_capacity),
        }
    }

    fn snapshot(&self) -> GenerationSnapshot<S> {
        GenerationSnapshot {
            id: self.id,
            model: self.model.clone(),
            deadline: self.deadline,
            write_stall_timeout: self.write_stall_timeout,
            service: self.service.clone(),
        }
    }
}

struct GenerationSnapshot<S> {
    id: u64,
    model: String,
    deadline: Duration,
    write_stall_timeout: Duration,
    service: ServiceActor<S>,
}

impl<S: ChatService> HttpState<S> {
    fn snapshot(&self) -> Result<GenerationSnapshot<S>, &'static str> {
        self.generation
            .read()
            .map(|generation| generation.snapshot())
            .map_err(|_| "runtime state failed")
    }
}

async fn live() -> Response {
    (StatusCode::OK, "live\n").into_response()
}

async fn ready<S>(State(state): State<Arc<HttpState<S>>>) -> Response {
    if state.ready.load(Ordering::Acquire) {
        (StatusCode::OK, "ready\n").into_response()
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "not ready\n").into_response()
    }
}

async fn prometheus<S: ChatService>(State(state): State<Arc<HttpState<S>>>) -> Response {
    let ready = state.ready.load(Ordering::Acquire);
    let metrics = state.snapshot().map_or_else(
        |_| ServingMetrics::default(),
        |runtime| runtime.service.metrics(),
    );
    let text = metrics.prometheus(true, ready);
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        text,
    )
        .into_response()
}

/// Stops admission, cancels active generation, and waits for its finalizer.
pub struct DrainHandle<S> {
    state: Arc<HttpState<S>>,
}

impl<S> Clone for DrainHandle<S> {
    fn clone(&self) -> Self {
        Self {
            state: Arc::clone(&self.state),
        }
    }
}

impl<S> DrainHandle<S> {
    /// Drains all admitted serving work.
    pub async fn drain(&self) {
        self.state.closed.store(true, Ordering::Release);
        self.state.ready.store(false, Ordering::Release);
        if let Ok(active) = self.state.active.lock() {
            for request in active.values() {
                request.control.cancel();
            }
        }
        loop {
            let notified = self.state.drained.notified();
            let done = self
                .state
                .active
                .lock()
                .map_or(true, |active| active.is_empty());
            if done {
                return;
            }
            notified.await;
        }
    }
}

impl<S: ChatService> DrainHandle<S> {
    /// Makes admission return service-unavailable responses.
    pub fn set_ready(&self, ready: bool) {
        if !self.state.closed.load(Ordering::Acquire) {
            self.state.ready.store(ready, Ordering::Release);
        }
    }

    /// Publishes a fully built serving generation atomically.
    ///
    /// # Errors
    ///
    /// Returns an error when the runtime lock is poisoned.
    pub fn publish(&self, generation: u64, service: S) -> Result<(), &'static str> {
        let next = PublishedGeneration::new(generation, service);
        let mut active = self
            .state
            .generation
            .write()
            .map_err(|_| "runtime state failed")?;
        *active = next;
        Ok(())
    }

    /// Waits for one generation, cancelling its remaining controls at the deadline.
    pub async fn drain_generation(&self, generation: u64, deadline: Duration) {
        self.set_ready(false);
        let expired = tokio::time::sleep(deadline);
        tokio::pin!(expired);
        loop {
            let notified = self.state.drained.notified();
            let remaining = self.state.active.lock().map_or(0, |active| {
                active
                    .values()
                    .filter(|request| request.generation == generation)
                    .count()
            });
            if remaining == 0 {
                return;
            }
            tokio::select! {
                () = &mut expired => break,
                () = notified => {}
            }
        }
        if let Ok(active) = self.state.active.lock() {
            for request in active
                .values()
                .filter(|request| request.generation == generation)
            {
                request.control.cancel();
            }
        }
        loop {
            let notified = self.state.drained.notified();
            let done = self.state.active.lock().map_or(true, |active| {
                active
                    .values()
                    .all(|request| request.generation != generation)
            });
            if done {
                return;
            }
            notified.await;
        }
    }
}

struct ActiveGuard<S> {
    state: Arc<HttpState<S>>,
    request_id: u64,
}

impl<S> Drop for ActiveGuard<S> {
    fn drop(&mut self) {
        if let Ok(mut active) = self.state.active.lock() {
            active.remove(&self.request_id);
        }
        self.state.drained.notify_waiters();
    }
}

type Body = Result<Bytes, BytesRejection>;

async fn body(request: Request) -> Body {
    Bytes::from_request(request, &()).await
}

async fn chat<S: ChatService>(
    State(state): State<Arc<HttpState<S>>>,
    request: Request,
) -> Response {
    let request_id = next_request_id();
    let runtime = match state.snapshot() {
        Ok(runtime) => runtime,
        Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let deadline = tokio::time::Instant::now() + runtime.deadline;
    let body = match body(request).await {
        Ok(body) => body,
        Err(error) => return body_error(&error),
    };
    let request = match serde_json::from_slice::<ChatCompletionRequest>(&body) {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    if request.model != runtime.model {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    if request.params.stream {
        return stream_chat_response(state, runtime, request, request_id, deadline).await;
    }
    let model = request.model.clone();
    let control = GenerationControl::new(runtime.write_stall_timeout);
    let active = match activate(&state, runtime.id, request_id, &control) {
        Ok(active) => active,
        Err(response) => return *response,
    };
    let actor = match runtime.service.complete(request, control.clone()) {
        Ok(receiver) => receiver,
        Err(error) => return actor_error(error),
    };
    let (sender, receiver) = oneshot::channel();
    let task_control = control.clone();
    tokio::spawn(async move {
        let _active = active;
        let generation = actor_result(actor);
        controlled_deadline(
            deadline,
            &task_control,
            generation,
            move |result| async move {
                let _ = sender.send(result);
            },
        )
        .await;
    });
    let mut disconnect = CancelOnDrop(Some(control));
    let Ok(result) = receiver.await else {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, "generation task failed");
    };
    disconnect.0 = None;
    let completion = match result {
        DeadlineResult::Completed(Ok(completion)) => completion,
        DeadlineResult::Completed(Err(error)) => return generation_error(&error),
        DeadlineResult::TimedOut => {
            return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded");
        }
    };
    log_completion(
        runtime.id,
        request_id,
        "/v1/chat/completions",
        &model,
        &completion,
    );
    let created = unix_time();
    match chat_response(&format!("chatcmpl-{created}"), created, &model, &completion) {
        Ok(body) => Json(body).into_response(),
        Err(error) => generation_error(&error),
    }
}

async fn text<S: ChatService>(
    State(state): State<Arc<HttpState<S>>>,
    request: Request,
) -> Response {
    let request_id = next_request_id();
    let runtime = match state.snapshot() {
        Ok(runtime) => runtime,
        Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    let deadline = tokio::time::Instant::now() + runtime.deadline;
    let body = match body(request).await {
        Ok(body) => body,
        Err(error) => return body_error(&error),
    };
    let request = match serde_json::from_slice::<CompletionRequest>(&body) {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    if request.model != runtime.model {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    if request.params.stream {
        return stream_text_response(&state, &runtime, request, request_id, deadline);
    }
    let model = request.model.clone();
    let control = GenerationControl::new(runtime.write_stall_timeout);
    let active = match activate(&state, runtime.id, request_id, &control) {
        Ok(active) => active,
        Err(response) => return *response,
    };
    let actor = match runtime.service.complete_text(request, control.clone()) {
        Ok(receiver) => receiver,
        Err(error) => return actor_error(error),
    };
    let (sender, receiver) = oneshot::channel();
    let task_control = control.clone();
    tokio::spawn(async move {
        let _active = active;
        let generation = actor_result(actor);
        controlled_deadline(
            deadline,
            &task_control,
            generation,
            move |result| async move {
                let _ = sender.send(result);
            },
        )
        .await;
    });
    let mut disconnect = CancelOnDrop(Some(control));
    let Ok(result) = receiver.await else {
        return api_error(StatusCode::INTERNAL_SERVER_ERROR, "generation task failed");
    };
    disconnect.0 = None;
    let completion = match result {
        DeadlineResult::Completed(Ok(completion)) => completion,
        DeadlineResult::Completed(Err(error)) => return generation_error(&error),
        DeadlineResult::TimedOut => {
            return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded");
        }
    };
    log_completion(
        runtime.id,
        request_id,
        "/v1/completions",
        &model,
        &completion,
    );
    let created = unix_time();
    match text_response(&format!("cmpl-{created}"), created, &model, &completion) {
        Ok(body) => Json(body).into_response(),
        Err(error) => generation_error(&error),
    }
}

async fn stream_chat_response<S: ChatService>(
    state: Arc<HttpState<S>>,
    runtime: GenerationSnapshot<S>,
    request: ChatCompletionRequest,
    request_id: u64,
    deadline: tokio::time::Instant,
) -> Response {
    if request.model != runtime.model {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    let model = request.model.clone();
    let created = unix_time();
    let id = format!("chatcmpl-{created}");
    let (sender, receiver) = mpsc::channel(16);
    if sender.send(GenerationEvent::Role).await.is_err() {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "cannot start response stream",
        );
    }
    let control = GenerationControl::new(runtime.write_stall_timeout);
    let active = match activate(&state, runtime.id, request_id, &control) {
        Ok(active) => active,
        Err(response) => return *response,
    };
    let actor = match runtime
        .service
        .stream_chat(request, sender.clone(), control.clone())
    {
        Ok(receiver) => receiver,
        Err(error) => return actor_error(error),
    };
    tokio::spawn(async move {
        let _active = active;
        let generation = actor_result(actor);
        controlled_deadline(deadline, &control, generation, move |result| async move {
            send_stream_error(result, &sender).await;
        })
        .await;
    });
    sse_response(receiver, id, created, model, true)
}

fn stream_text_response<S: ChatService>(
    state: &Arc<HttpState<S>>,
    runtime: &GenerationSnapshot<S>,
    request: CompletionRequest,
    request_id: u64,
    deadline: tokio::time::Instant,
) -> Response {
    if request.model != runtime.model {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    let model = request.model.clone();
    let created = unix_time();
    let id = format!("cmpl-{created}");
    let (sender, receiver) = mpsc::channel(16);
    let control = GenerationControl::new(runtime.write_stall_timeout);
    let active = match activate(state, runtime.id, request_id, &control) {
        Ok(active) => active,
        Err(response) => return *response,
    };
    let actor = match runtime
        .service
        .stream_text(request, sender.clone(), control.clone())
    {
        Ok(receiver) => receiver,
        Err(error) => return actor_error(error),
    };
    tokio::spawn(async move {
        let _active = active;
        let generation = actor_result(actor);
        controlled_deadline(deadline, &control, generation, move |result| async move {
            send_stream_error(result, &sender).await;
        })
        .await;
    });
    sse_response(receiver, id, created, model, false)
}

enum DeadlineResult<T> {
    Completed(Result<T, GenerationError>),
    TimedOut,
}

async fn actor_result<T>(
    receiver: oneshot::Receiver<Result<T, GenerationError>>,
) -> Result<T, GenerationError> {
    receiver
        .await
        .map_err(|_| GenerationError::new("serving scheduler stopped"))?
}

struct CancelOnDrop(Option<GenerationControl>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if let Some(control) = &self.0 {
            control.cancel();
        }
    }
}

async fn controlled_deadline<T, F, N, NF>(
    deadline: tokio::time::Instant,
    control: &GenerationControl,
    future: F,
    notify: N,
) where
    F: Future<Output = Result<T, GenerationError>>,
    N: FnOnce(DeadlineResult<T>) -> NF,
    NF: Future<Output = ()>,
{
    let timer = tokio::time::sleep_until(deadline);
    let mut future = std::pin::pin!(future);
    let mut timer = std::pin::pin!(timer);
    let completed = std::future::poll_fn(|context| {
        if let std::task::Poll::Ready(result) = future.as_mut().poll(context) {
            return std::task::Poll::Ready(Some(result));
        }
        if timer.as_mut().poll(context).is_ready() {
            return std::task::Poll::Ready(None);
        }
        std::task::Poll::Pending
    })
    .await;
    if let Some(result) = completed {
        notify(DeadlineResult::Completed(result)).await;
    } else {
        control.cancel();
        notify(DeadlineResult::TimedOut).await;
        let _ = future.await;
    }
}

async fn send_stream_error(result: DeadlineResult<()>, sender: &mpsc::Sender<GenerationEvent>) {
    let message = match result {
        DeadlineResult::Completed(Ok(())) => return,
        DeadlineResult::Completed(Err(error)) => error.to_string(),
        DeadlineResult::TimedOut => "request deadline exceeded".to_owned(),
    };
    let _ = sender.send(GenerationEvent::Error(message)).await;
}

fn sse_response(
    receiver: mpsc::Receiver<GenerationEvent>,
    id: String,
    created: u64,
    model: String,
    chat: bool,
) -> Response {
    let stream = ReceiverStream::new(receiver).map(move |event| {
        Ok::<Event, Infallible>(
            Event::default().data(stream_data(event, &id, created, &model, chat)),
        )
    });
    Sse::new(stream).into_response()
}

fn stream_data(event: GenerationEvent, id: &str, created: u64, model: &str, chat: bool) -> String {
    let object = if chat {
        "chat.completion.chunk"
    } else {
        "text_completion"
    };
    match event {
        GenerationEvent::Role => json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": Value::Null}],
        })
        .to_string(),
        GenerationEvent::Content(text) if chat => json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": Value::Null}],
        })
        .to_string(),
        GenerationEvent::Content(text) => json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [{"index": 0, "text": text, "finish_reason": Value::Null}],
        })
        .to_string(),
        GenerationEvent::ToolCallStart(name) if chat => json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "id": "call_0", "type": "function",
                "function": {"name": name, "arguments": ""}
            }]}, "finish_reason": Value::Null}],
        })
        .to_string(),
        GenerationEvent::ToolArguments(arguments) if chat => json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [{"index": 0, "delta": {"tool_calls": [{
                "index": 0, "function": {"arguments": arguments}
            }]}, "finish_reason": Value::Null}],
        })
        .to_string(),
        GenerationEvent::ToolCallStart(_) | GenerationEvent::ToolArguments(_) => {
            error_value("text completions cannot emit tool calls", "server_error").to_string()
        }
        GenerationEvent::Finished(reason, _) if chat => json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": crate::response::finish_reason(reason)}],
        })
        .to_string(),
        GenerationEvent::Finished(reason, _) => json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [{"index": 0, "text": "", "finish_reason": crate::response::finish_reason(reason)}],
        })
        .to_string(),
        GenerationEvent::Usage(usage) => json!({
            "id": id, "object": object, "created": created, "model": model,
            "choices": [], "usage": stream_usage(usage),
        })
        .to_string(),
        GenerationEvent::Done => "[DONE]".to_owned(),
        GenerationEvent::Error(message) => error_value(&message, "server_error").to_string(),
    }
}

fn stream_usage(usage: Usage) -> Value {
    json!({
        "prompt_tokens": usage.prompt_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_tokens},
        "completion_tokens": usage.completion_tokens,
        "total_tokens": u64::from(usage.prompt_tokens) + u64::from(usage.completion_tokens),
    })
}

async fn models<S: ChatService>(State(state): State<Arc<HttpState<S>>>) -> Response {
    match state.snapshot() {
        Ok(runtime) => {
            Json(json!({"object": "list", "data": [model_object(&runtime.model)]})).into_response()
        }
        Err(error) => api_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    }
}

async fn model<S: ChatService>(
    State(state): State<Arc<HttpState<S>>>,
    Path(requested): Path<String>,
) -> Response {
    let runtime = match state.snapshot() {
        Ok(runtime) => runtime,
        Err(error) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, error),
    };
    if requested == runtime.model {
        Json(model_object(&runtime.model)).into_response()
    } else {
        api_error(StatusCode::NOT_FOUND, "model is not loaded")
    }
}

fn model_object(model: &str) -> Value {
    json!({"id": model, "object": "model", "created": 0, "owned_by": "forja"})
}

fn unix_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs())
}

fn next_request_id() -> u64 {
    NEXT_REQUEST_ID
        .try_update(Ordering::AcqRel, Ordering::Acquire, |id| id.checked_add(1))
        .unwrap_or(u64::MAX)
}

fn log_completion(
    config_generation: u64,
    request_id: u64,
    route: &str,
    model: &str,
    completion: &Completion,
) {
    tracing::info!(
        event = "request.completed",
        request_id,
        config_generation,
        model,
        route,
        outcome = "ok",
        status = 200,
        finish_reason = crate::response::finish_reason(completion.finish_reason),
        prompt_tokens = completion.usage.prompt_tokens,
        output_tokens = completion.usage.completion_tokens,
        cached_tokens = completion.usage.cached_tokens,
    );
}

fn activate<S>(
    state: &Arc<HttpState<S>>,
    generation: u64,
    request_id: u64,
    control: &GenerationControl,
) -> Result<ActiveGuard<S>, Box<Response>> {
    let mut active = state.active.lock().map_err(|_| {
        Box::new(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "admission state failed",
        ))
    })?;
    if state.closed.load(Ordering::Acquire) || !state.ready.load(Ordering::Acquire) {
        return Err(Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "model is unavailable",
        )));
    }
    active.insert(
        request_id,
        ActiveRequest {
            generation,
            control: control.clone(),
        },
    );
    drop(active);
    Ok(ActiveGuard {
        state: Arc::clone(state),
        request_id,
    })
}

fn actor_error(error: ActorError) -> Response {
    match error {
        ActorError::Full => api_error(StatusCode::TOO_MANY_REQUESTS, "request queue is full"),
        ActorError::Closed => api_error(StatusCode::SERVICE_UNAVAILABLE, "server is draining"),
    }
}

fn body_error(error: &BytesRejection) -> Response {
    if error.status() == StatusCode::PAYLOAD_TOO_LARGE {
        api_error(
            StatusCode::PAYLOAD_TOO_LARGE,
            "request body exceeds the configured ceiling",
        )
    } else {
        api_error(StatusCode::BAD_REQUEST, &error.to_string())
    }
}

fn generation_error(error: &GenerationError) -> Response {
    match error.kind() {
        ErrorKind::Invalid => api_error(StatusCode::BAD_REQUEST, &error.to_string()),
        ErrorKind::PayloadTooLarge => api_error(StatusCode::PAYLOAD_TOO_LARGE, &error.to_string()),
        ErrorKind::Internal => api_error(StatusCode::INTERNAL_SERVER_ERROR, &error.to_string()),
        ErrorKind::Unavailable => api_error(StatusCode::SERVICE_UNAVAILABLE, &error.to_string()),
    }
}

fn api_error(status: StatusCode, message: &str) -> Response {
    let error_type = if status.is_client_error() {
        "invalid_request_error"
    } else {
        "server_error"
    };
    (status, Json(error_value(message, error_type))).into_response()
}

fn error_value(message: &str, error_type: &str) -> Value {
    json!({"error": {
        "message": message,
        "type": error_type,
        "param": Value::Null,
        "code": Value::Null,
    }})
}

#[cfg(test)]
mod tests {
    use std::{
        error::Error,
        fs,
        io::Write,
        path::PathBuf,
        sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use forja_host::{EngineLoadConfig, EngineRunner, Limits};
    use tokenizers::{Tokenizer, decoders::byte_level::ByteLevel, models::bpe::BPE};
    use tracing_subscriber::fmt::MakeWriter;

    use super::*;
    use crate::{FinishReason, Usage};

    struct Smoke;

    impl ChatService for Smoke {
        fn model(&self) -> &'static str {
            "smoke"
        }

        fn max_body_bytes(&self) -> usize {
            1024
        }

        fn queue_length(&self) -> u32 {
            1
        }

        fn request_deadline(&self) -> Duration {
            Duration::from_secs(1)
        }

        fn write_stall_timeout(&self) -> Duration {
            Duration::from_secs(1)
        }

        fn complete(
            &mut self,
            _: ChatCompletionRequest,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            Box::pin(async {
                Ok(Completion {
                    text: "forged".to_owned(),
                    tool_call: None,
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        prompt_tokens: 4,
                        cached_tokens: 0,
                        completion_tokens: 1,
                    },
                    token_ids: vec![1],
                })
            })
        }

        fn complete_text(
            &mut self,
            _: CompletionRequest,
            control: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            self.complete(
                ChatCompletionRequest {
                    model: "smoke".to_owned(),
                    messages: Vec::new(),
                    params: crate::CompletionParams::default(),
                },
                control,
            )
        }

        fn stream_chat(
            &mut self,
            _: ChatCompletionRequest,
            sender: mpsc::Sender<GenerationEvent>,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
            Box::pin(async move {
                sender
                    .send(GenerationEvent::Content("forged".to_owned()))
                    .await
                    .unwrap();
                sender
                    .send(GenerationEvent::Finished(
                        FinishReason::Stop,
                        Usage {
                            prompt_tokens: 4,
                            cached_tokens: 0,
                            completion_tokens: 1,
                        },
                    ))
                    .await
                    .unwrap();
                sender.send(GenerationEvent::Done).await.unwrap();
                Ok(())
            })
        }

        fn stream_text(
            &mut self,
            _: CompletionRequest,
            sender: mpsc::Sender<GenerationEvent>,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
            Box::pin(async move {
                sender.send(GenerationEvent::Done).await.unwrap();
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn service_actor_owns_requests_and_reports_queue_time() {
        let actor = ServiceActor::spawn(Smoke, 1);
        let request = ChatCompletionRequest {
            model: "smoke".to_owned(),
            messages: Vec::new(),
            params: crate::CompletionParams::default(),
        };
        let receiver = actor
            .complete(request, GenerationControl::new(Duration::from_secs(1)))
            .unwrap();
        let completion = receiver.await.unwrap().unwrap();
        assert_eq!(completion.text, "forged");
        tokio::task::yield_now().await;
        let metrics = actor.metrics().prometheus(true, true);
        assert!(metrics.contains("forja_queue_seconds_count 1\n"));
        assert!(metrics.contains("forja_scheduler_seconds_count 1\n"));
    }

    struct DeadlineSmoke {
        release: Arc<tokio::sync::Notify>,
        block_next: bool,
    }

    impl ChatService for DeadlineSmoke {
        fn model(&self) -> &'static str {
            "smoke"
        }

        fn max_body_bytes(&self) -> usize {
            1024
        }

        fn queue_length(&self) -> u32 {
            1
        }

        fn request_deadline(&self) -> Duration {
            Duration::from_millis(30)
        }

        fn write_stall_timeout(&self) -> Duration {
            Duration::from_secs(1)
        }

        fn complete(
            &mut self,
            _: ChatCompletionRequest,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            let block = self.block_next;
            self.block_next = false;
            let release = Arc::clone(&self.release);
            Box::pin(async move {
                if block {
                    release.notified().await;
                }
                Ok(Completion {
                    text: "forged".to_owned(),
                    tool_call: None,
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        prompt_tokens: 4,
                        cached_tokens: 0,
                        completion_tokens: 1,
                    },
                    token_ids: vec![1],
                })
            })
        }

        fn complete_text(
            &mut self,
            _: CompletionRequest,
            control: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            self.complete(
                ChatCompletionRequest {
                    model: "smoke".to_owned(),
                    messages: Vec::new(),
                    params: crate::CompletionParams::default(),
                },
                control,
            )
        }

        fn stream_chat(
            &mut self,
            _: ChatCompletionRequest,
            _: mpsc::Sender<GenerationEvent>,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
            Box::pin(async { Ok(()) })
        }

        fn stream_text(
            &mut self,
            _: CompletionRequest,
            _: mpsc::Sender<GenerationEvent>,
            _: GenerationControl,
        ) -> Pin<Box<dyn Future<Output = Result<(), GenerationError>> + Send + '_>> {
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    async fn handles_one_chat_request() {
        let body = br#"{"model":"smoke","messages":[{"role":"user","content":"hi"}]}"#;
        let response = chat::<Smoke>(State(state()), request(body)).await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn serves_text_completions_and_model_metadata() {
        let state = state();
        let response = text::<Smoke>(
            State(Arc::clone(&state)),
            request(br#"{"model":"smoke","prompt":"hi"}"#),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            models::<Smoke>(State(Arc::clone(&state))).await.status(),
            StatusCode::OK
        );
        assert_eq!(
            model::<Smoke>(State(state), Path("missing".to_owned()))
                .await
                .status(),
            StatusCode::NOT_FOUND
        );
    }

    #[tokio::test]
    async fn health_and_metrics_reflect_readiness() {
        let state = state();
        assert_eq!(live().await.status(), StatusCode::OK);
        assert_eq!(
            ready::<Smoke>(State(Arc::clone(&state))).await.status(),
            StatusCode::OK
        );
        let metrics = prometheus::<Smoke>(State(Arc::clone(&state))).await;
        assert_eq!(metrics.status(), StatusCode::OK);
        assert_eq!(
            metrics.headers()[header::CONTENT_TYPE],
            "text/plain; version=0.0.4; charset=utf-8"
        );
        state.ready.store(false, Ordering::Release);
        assert_eq!(
            ready::<Smoke>(State(state)).await.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn bounds_the_fifo_waiting_queue() {
        let release = Arc::new(tokio::sync::Notify::new());
        let actor = ServiceActor::spawn(
            DeadlineSmoke {
                release: Arc::clone(&release),
                block_next: true,
            },
            1,
        );
        let first = actor.complete(chat_request(), control()).unwrap();
        tokio::task::yield_now().await;
        let second = actor.complete(chat_request(), control()).unwrap();
        assert_eq!(
            actor.complete(chat_request(), control()).unwrap_err(),
            ActorError::Full
        );
        release.notify_one();
        assert!(first.await.unwrap().is_ok());
        assert!(second.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn wrong_model_never_enters_admission() {
        let state = state();
        let chat_request = br#"{"model":"missing","messages":[]}"#;
        let text_request = br#"{"model":"missing","prompt":"hi"}"#;

        let chat_response = chat::<Smoke>(State(Arc::clone(&state)), request(chat_request)).await;
        let text_response = text::<Smoke>(State(Arc::clone(&state)), request(text_request)).await;
        let stream_chat: ChatCompletionRequest = serde_json::from_slice(chat_request).unwrap();
        let stream_text: CompletionRequest = serde_json::from_slice(text_request).unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        let stream_chat_response = stream_chat_response(
            Arc::clone(&state),
            state.snapshot().unwrap(),
            stream_chat,
            1,
            deadline,
        )
        .await;
        let runtime = state.snapshot().unwrap();
        let stream_text_response = stream_text_response(&state, &runtime, stream_text, 2, deadline);

        assert_eq!(chat_response.status(), StatusCode::NOT_FOUND);
        assert_eq!(text_response.status(), StatusCode::NOT_FOUND);
        assert_eq!(stream_chat_response.status(), StatusCode::NOT_FOUND);
        assert_eq!(stream_text_response.status(), StatusCode::NOT_FOUND);
        assert!(state.active.lock().unwrap().is_empty());
    }

    #[test]
    fn serializes_stream_chunks_and_clean_termination() {
        let usage = Usage {
            prompt_tokens: 7,
            cached_tokens: 3,
            completion_tokens: 3,
        };
        let role = stream_data(GenerationEvent::Role, "id", 1, "model", true);
        let content = stream_data(
            GenerationEvent::Content("hi".to_owned()),
            "id",
            1,
            "model",
            true,
        );
        let finish = stream_data(
            GenerationEvent::Finished(FinishReason::Stop, usage),
            "id",
            1,
            "model",
            true,
        );
        let call = stream_data(
            GenerationEvent::ToolCallStart("read_file".to_owned()),
            "id",
            1,
            "model",
            true,
        );
        let arguments = stream_data(
            GenerationEvent::ToolArguments("{\"path\":\"a.rs\"}".to_owned()),
            "id",
            1,
            "model",
            true,
        );
        let usage = stream_data(GenerationEvent::Usage(usage), "id", 1, "model", true);
        assert!(role.contains("assistant"));
        assert!(content.contains("hi"));
        assert!(finish.contains("stop"));
        assert!(call.contains("call_0"));
        assert!(call.contains("read_file"));
        assert!(arguments.contains("path"));
        assert!(usage.contains("total_tokens\":10"));
        assert!(usage.contains("cached_tokens\":3"));
        assert_eq!(
            stream_data(GenerationEvent::Done, "id", 1, "model", true),
            "[DONE]"
        );
    }

    #[test]
    fn structured_completion_event_omits_generated_content() {
        #[derive(Clone)]
        struct Capture(Arc<StdMutex<Vec<u8>>>);

        impl Write for Capture {
            fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().write(bytes)
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        impl<'a> MakeWriter<'a> for Capture {
            type Writer = Self;

            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let output = Arc::new(StdMutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(Capture(Arc::clone(&output)))
            .finish();
        let completion = Completion {
            text: "secret prompt completion and tool body".to_owned(),
            tool_call: None,
            finish_reason: FinishReason::Stop,
            usage: Usage {
                prompt_tokens: 7,
                cached_tokens: 3,
                completion_tokens: 2,
            },
            token_ids: vec![1, 2],
        };
        tracing::subscriber::with_default(subscriber, || {
            log_completion(3, 7, "/v1/completions", "model", &completion);
        });
        let output = String::from_utf8(output.lock().unwrap().clone()).unwrap();
        assert!(output.contains("request.completed"));
        assert!(output.contains("prompt_tokens"));
        assert!(!output.contains("secret prompt completion and tool body"));
    }

    #[test]
    fn declared_tool_byte_excess_is_payload_too_large() {
        assert_eq!(
            generation_error(&GenerationError::too_large("tool bytes")).status(),
            StatusCode::PAYLOAD_TOO_LARGE
        );
    }

    #[tokio::test]
    async fn deadline_and_disconnect_cancel_idempotently() {
        let control = GenerationControl::new(Duration::from_secs(1));
        let observed = control.clone();
        let (sender, receiver) = oneshot::channel();
        controlled_deadline(
            tokio::time::Instant::now(),
            &control,
            async move {
                while !observed.is_cancelled() {
                    tokio::task::yield_now().await;
                }
                Ok(())
            },
            move |result| async move {
                let _ = sender.send(matches!(result, DeadlineResult::TimedOut));
            },
        )
        .await;
        assert!(receiver.await.unwrap());

        let disconnected = GenerationControl::new(Duration::from_secs(1));
        drop(CancelOnDrop(Some(disconnected.clone())));
        drop(CancelOnDrop(Some(disconnected.clone())));
        assert!(disconnected.is_cancelled());
    }

    #[tokio::test]
    async fn deadline_replies_before_finalization_and_retains_admission() {
        let release = Arc::new(tokio::sync::Notify::new());
        let state = test_state(DeadlineSmoke {
            release: Arc::clone(&release),
            block_next: true,
        });
        let body = br#"{"model":"smoke","messages":[]}"#;
        let started = tokio::time::Instant::now();

        let first = chat::<DeadlineSmoke>(State(Arc::clone(&state)), request(body)).await;

        assert_eq!(first.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(started.elapsed() < Duration::from_millis(250));
        assert_eq!(state.active.lock().unwrap().len(), 1);
        let next = tokio::spawn(chat::<DeadlineSmoke>(
            State(Arc::clone(&state)),
            request(body),
        ));
        tokio::task::yield_now().await;
        assert!(!next.is_finished());
        release.notify_one();
        let next = tokio::time::timeout(Duration::from_secs(1), next)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(next.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn drain_stops_admission_and_waits_for_finalization() {
        let state = state();
        let control = GenerationControl::new(Duration::from_secs(1));
        let active = activate(&state, 0, 1, &control).unwrap();
        assert_eq!(state.active.lock().unwrap()[&1].generation, 0);
        let handle = DrainHandle {
            state: Arc::clone(&state),
        };
        let drain = handle.drain();
        tokio::pin!(drain);
        assert!(matches!(
            futures_poll_once(drain.as_mut()),
            std::task::Poll::Pending
        ));
        assert!(control.is_cancelled());
        drop(active);
        drain.await;
        assert_eq!(
            activate(&state, 0, 2, &control).err().unwrap().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    #[tokio::test]
    async fn publishes_a_complete_runtime_generation() {
        let state = state();
        let handle = DrainHandle {
            state: Arc::clone(&state),
        };
        handle.publish(7, Smoke).unwrap();
        assert_eq!(state.generation.read().unwrap().id, 7);
    }

    #[tokio::test]
    async fn generation_deadline_cancels_only_the_drained_runtime() {
        let state = state();
        let handle = DrainHandle {
            state: Arc::clone(&state),
        };
        let old = GenerationControl::new(Duration::from_secs(1));
        let next = GenerationControl::new(Duration::from_secs(1));
        let old_guard = activate(&state, 0, 1, &old).unwrap();
        let next_guard = activate(&state, 1, 2, &next).unwrap();
        let drain = tokio::spawn(async move {
            handle.drain_generation(0, Duration::ZERO).await;
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while !old.is_cancelled() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert!(!next.is_cancelled());
        drop(old_guard);
        drain.await.unwrap();
        drop(next_guard);
    }

    fn state() -> Arc<HttpState<Smoke>> {
        test_state(Smoke)
    }

    fn test_state<S: ChatService>(service: S) -> Arc<HttpState<S>> {
        Arc::new(HttpState {
            generation: RwLock::new(PublishedGeneration::new(0, service)),
            ready: std::sync::atomic::AtomicBool::new(true),
            closed: std::sync::atomic::AtomicBool::new(false),
            active: StdMutex::new(BTreeMap::new()),
            drained: tokio::sync::Notify::new(),
        })
    }

    fn chat_request() -> ChatCompletionRequest {
        ChatCompletionRequest {
            model: "smoke".to_owned(),
            messages: Vec::new(),
            params: crate::CompletionParams::default(),
        }
    }

    fn control() -> GenerationControl {
        GenerationControl::new(Duration::from_secs(1))
    }

    fn request(bytes: &'static [u8]) -> Request {
        Request::builder()
            .body(axum::body::Body::from(bytes))
            .unwrap()
    }

    fn futures_poll_once<F: Future>(future: Pin<&mut F>) -> std::task::Poll<F::Output> {
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        future.poll(&mut context)
    }

    type ContinuousActor = ServiceActor<crate::ServingEngine<forja_cpu::CpuBackend>>;
    static NEXT_WEIGHTS: AtomicU64 = AtomicU64::new(0);

    struct TestWeights {
        directory: PathBuf,
        path: PathBuf,
    }

    impl Drop for TestWeights {
        fn drop(&mut self) {
            drop(fs::remove_dir_all(&self.directory));
        }
    }

    async fn continuous_actor()
    -> Result<(ContinuousActor, TestWeights), Box<dyn Error + Send + Sync>> {
        continuous_actor_with_barrier(None).await
    }

    async fn continuous_actor_with_barrier(
        barrier: Option<Arc<crate::engine::BatchTestBarrier>>,
    ) -> Result<(ContinuousActor, TestWeights), Box<dyn Error + Send + Sync>> {
        let weights = test_weights()?;
        let mut runner = EngineRunner::new(
            test_guests::engine_smoke(),
            forja_cpu::CpuBackend::new(),
            Limits::new(1024 * 1024, 4, 1024, 128, 1024 * 1024),
            &weights.path,
        )
        .await?;
        let info = runner.describe().await?;
        runner
            .load_with_selections(
                None,
                &EngineLoadConfig {
                    paged_cache: true,
                    ..EngineLoadConfig::default()
                },
            )
            .await??;
        let vocab = [
            ("a".to_owned(), 0),
            ("b".to_owned(), 1),
            ("c".to_owned(), 2),
            ("d".to_owned(), 3),
        ];
        let model = BPE::builder().vocab_and_merges(vocab, Vec::new()).build()?;
        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_decoder(Some(ByteLevel::default()));
        let chat_format = crate::ChatFormat::load(
            test_guests::continuous_chat_format(),
            crate::ChatFormatLimits::default(),
        )?;
        let mut config = forja_config::HostConfig::default();
        config.server.queue_length = 16;
        let mut engine = crate::ServingEngine::new(
            "smoke".to_owned(),
            runner,
            tokenizer,
            chat_format,
            config,
            info,
            32,
            16,
            crate::PrefixIdentity::new(["engine", "profile", "weights", "tokenizer", "chat"])?,
        )?;
        if let Some(barrier) = barrier {
            engine.set_batch_test_barrier(barrier);
        }
        Ok((ServiceActor::spawn(engine, 16), weights))
    }

    fn text_request(prompt: &str, max_tokens: u32) -> CompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "smoke",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0
        }))
        .unwrap()
    }

    fn forced_tool_request(max_tokens: u32) -> ChatCompletionRequest {
        serde_json::from_value(serde_json::json!({
            "model": "smoke",
            "messages": [{"role": "user", "content": "a"}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "tools": [{
                "type": "function",
                "function": {
                    "name": "tool",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": false
                    }
                }
            }],
            "tool_choice": "required"
        }))
        .unwrap()
    }

    async fn actor_completion(
        actor: &ContinuousActor,
        prompt: &str,
        max_tokens: u32,
    ) -> Result<Completion, GenerationError> {
        actor
            .complete_text(
                text_request(prompt, max_tokens),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap()
            .await
            .unwrap()
    }

    async fn serial_output(prompt: &str, max_tokens: u32) -> (Vec<u8>, Vec<u32>) {
        let (actor, _weights) = continuous_actor().await.unwrap();
        let completion = actor_completion(&actor, prompt, max_tokens).await.unwrap();
        (
            completion.text.as_bytes().to_vec(),
            completion.token_ids().to_vec(),
        )
    }

    fn assert_serial(completion: &Completion, expected: &(Vec<u8>, Vec<u32>)) {
        assert_eq!(completion.text.as_bytes(), expected.0);
        assert_eq!(completion.token_ids(), expected.1);
    }

    #[tokio::test]
    async fn cancelling_one_submitted_lane_preserves_its_peer() {
        let expected = serial_output("b", 6).await;
        let (actor, _weights) = continuous_actor().await.unwrap();
        let control = GenerationControl::new(Duration::from_secs(2));
        let (events, mut received) = mpsc::channel(1);
        let cancelled = actor
            .stream_text(text_request("a", 8), events, control.clone())
            .unwrap();
        let survivor = actor
            .complete_text(
                text_request("b", 6),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        received.recv().await.unwrap();
        control.cancel();
        assert!(cancelled.await.unwrap().is_err());
        assert_serial(&survivor.await.unwrap().unwrap(), &expected);
    }

    #[tokio::test]
    async fn cancelled_lane_is_released_before_its_slot_is_reused() {
        let expected = serial_output("b", 6).await;
        let barrier = Arc::new(crate::engine::BatchTestBarrier::passive());
        let (actor, _weights) = continuous_actor_with_barrier(Some(Arc::clone(&barrier)))
            .await
            .unwrap();
        let retiring = actor
            .complete(
                forced_tool_request(1),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        let peer = actor
            .complete_text(
                text_request("a", 2),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        assert!(retiring.await.unwrap().is_ok());
        assert!(peer.await.unwrap().is_ok());
        assert!(barrier.deferred_fallthroughs() > 0);

        let control = GenerationControl::new(Duration::from_secs(2));
        barrier.cancel_on_next_batch(control.clone());
        let cancelled = actor
            .complete(forced_tool_request(16), control.clone())
            .unwrap();
        let survivor = actor
            .complete_text(
                text_request("b", 6),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        let replacement = actor
            .complete_text(
                text_request("a", 2),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();

        assert!(cancelled.await.unwrap().is_err());
        assert_serial(&survivor.await.unwrap().unwrap(), &expected);
        assert!(replacement.await.unwrap().is_ok());
    }

    #[tokio::test]
    async fn stale_lane_is_released_before_admission() {
        let barrier = Arc::new(crate::engine::BatchTestBarrier::stale_lane());
        let (actor, _weights) = continuous_actor_with_barrier(Some(barrier)).await.unwrap();

        assert!(actor_completion(&actor, "a", 2).await.is_ok());
    }

    #[tokio::test]
    async fn lane_retirement_does_not_change_an_inflight_successor() {
        let short_expected = serial_output("b", 2).await;
        let long_expected = serial_output("b", 6).await;
        let barrier = Arc::new(crate::engine::BatchTestBarrier::passive());
        let (actor, _weights) = continuous_actor_with_barrier(Some(Arc::clone(&barrier)))
            .await
            .unwrap();
        let short = actor
            .complete_text(
                text_request("b", 2),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        let long = actor
            .complete_text(
                text_request("b", 6),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        assert_serial(&short.await.unwrap().unwrap(), &short_expected);
        assert_serial(&long.await.unwrap().unwrap(), &long_expected);
        assert!(barrier.retirement_drained());
        assert!(barrier.prefill_overlap_width() >= 2);
    }

    #[tokio::test]
    async fn admission_during_a_successor_preserves_both_outputs() {
        let short_expected = serial_output("b", 6).await;
        let survivor_expected = serial_output("b", 8).await;
        let newcomer_expected = serial_output("a", 2).await;
        let barrier = Arc::new(crate::engine::BatchTestBarrier::new());
        let (actor, _weights) = continuous_actor_with_barrier(Some(Arc::clone(&barrier)))
            .await
            .unwrap();
        let short = actor
            .complete_text(
                text_request("b", 6),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        let survivor = actor
            .complete_text(
                text_request("b", 8),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        barrier.reached().await;
        let newcomer = actor
            .complete_text(
                text_request("a", 2),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        barrier.release();
        assert_serial(&short.await.unwrap().unwrap(), &short_expected);
        assert_serial(&survivor.await.unwrap().unwrap(), &survivor_expected);
        assert_serial(&newcomer.await.unwrap().unwrap(), &newcomer_expected);
        assert_eq!(survivor_expected.1.len(), 8);
        assert!(!barrier.admission_overlapped());
    }

    #[tokio::test]
    async fn lane_zero_survives_nonzero_lane_prefill() {
        let expected = serial_output("b", 6).await;
        let newcomer_expected = serial_output("a", 2).await;
        let barrier = Arc::new(crate::engine::BatchTestBarrier::passive());
        let (actor, _weights) = continuous_actor_with_barrier(Some(Arc::clone(&barrier)))
            .await
            .unwrap();
        let survivor = actor
            .complete_text(
                text_request("b", 6),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        tokio::task::yield_now().await;
        let newcomer = actor
            .complete_text(
                text_request("a", 2),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        assert_serial(&survivor.await.unwrap().unwrap(), &expected);
        assert_serial(&newcomer.await.unwrap().unwrap(), &newcomer_expected);
        assert!(barrier.prefill_overlap_width() >= 1);
    }

    #[tokio::test]
    async fn failed_batch_does_not_poison_later_lanes() {
        let expected = serial_output("b", 6).await;
        let (actor, _weights) = continuous_actor().await.unwrap();
        let first = actor
            .complete_text(
                text_request("a", 8),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        let peer = actor
            .complete_text(
                text_request("b", 8),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        assert!(first.await.unwrap().is_err());
        assert!(peer.await.unwrap().is_err());
        assert_serial(&actor_completion(&actor, "b", 6).await.unwrap(), &expected);
    }

    #[tokio::test]
    async fn constrained_generation_yields_to_ordinary_batches() {
        let expected = serial_output("b", 2).await;
        let barrier = Arc::new(crate::engine::BatchTestBarrier::new());
        let (actor, _weights) = continuous_actor_with_barrier(Some(Arc::clone(&barrier)))
            .await
            .unwrap();
        let constrained = actor
            .complete(
                forced_tool_request(16),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        barrier.reached().await;
        let ordinary = actor
            .complete_text(
                text_request("b", 2),
                GenerationControl::new(Duration::from_secs(2)),
            )
            .unwrap();
        barrier.release();

        assert_serial(&ordinary.await.unwrap().unwrap(), &expected);
        assert_eq!(
            constrained.await.unwrap().unwrap().finish_reason,
            FinishReason::Length
        );
        let (ordinary, constrained) = barrier.finish_order();
        assert!(
            ordinary > 0 && ordinary < constrained,
            "finish order was ordinary={ordinary}, constrained={constrained}"
        );
    }

    fn test_weights() -> Result<TestWeights, Box<dyn Error + Send + Sync>> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let nonce = NEXT_WEIGHTS.fetch_add(1, AtomicOrdering::Relaxed);
        let directory = std::env::temp_dir().join(format!(
            "forja-continuous-serving-{}-{timestamp}-{nonce}",
            std::process::id()
        ));
        fs::create_dir(&directory)?;
        let path = directory.join("weights.safetensors");
        fs::write(&path, empty_safetensors())?;
        Ok(TestWeights { directory, path })
    }

    fn empty_safetensors() -> Vec<u8> {
        let mut header = br#"{"value":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
        while !(header.len() + 8).is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend_from_slice(&0_f32.to_le_bytes());
        bytes
    }
}
