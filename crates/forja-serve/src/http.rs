use std::{
    convert::Infallible,
    future::Future,
    pin::Pin,
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicU32, Ordering},
    },
    time::{Duration, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, FromRequest, Path, Request, State, rejection::BytesRejection},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use forja_core::Backend;
use serde_json::{Value, json};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

use crate::engine::ErrorKind;
use crate::{
    ChatCompletionRequest, Completion, CompletionRequest, GenerationControl, GenerationError,
    GenerationEvent, ServingEngine, Usage, chat_response, text_response,
};

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

/// Builds the OpenAI-compatible chat-completions router.
pub fn chat_router<S: ChatService>(service: S) -> Router {
    serving_router(service).0
}

/// Builds a serving router and a handle that drains its admitted work.
pub fn serving_router<S: ChatService>(service: S) -> (Router, DrainHandle<S>) {
    let max_body_bytes = service.max_body_bytes();
    let served_model = service.model().to_owned();
    let state = Arc::new(HttpState {
        model: served_model,
        queue_limit: service.queue_length(),
        deadline: service.request_deadline(),
        write_stall_timeout: service.write_stall_timeout(),
        service: Mutex::new(service),
        sequence: Arc::new(Semaphore::new(1)),
        queued: AtomicU32::new(0),
        draining: std::sync::atomic::AtomicBool::new(false),
        active: StdMutex::new(None),
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

struct HttpState<S> {
    model: String,
    service: Mutex<S>,
    sequence: Arc<Semaphore>,
    queued: AtomicU32,
    draining: std::sync::atomic::AtomicBool,
    active: StdMutex<Option<GenerationControl>>,
    drained: tokio::sync::Notify,
    queue_limit: u32,
    deadline: Duration,
    write_stall_timeout: Duration,
}

/// Stops admission, cancels active generation, and waits for its finalizer.
pub struct DrainHandle<S> {
    state: Arc<HttpState<S>>,
}

impl<S> DrainHandle<S> {
    /// Drains all admitted serving work.
    pub async fn drain(&self) {
        self.state.draining.store(true, Ordering::Release);
        self.state.sequence.close();
        if let Ok(active) = self.state.active.lock()
            && let Some(control) = active.as_ref()
        {
            control.cancel();
        }
        loop {
            let notified = self.state.drained.notified();
            let done = self
                .state
                .active
                .lock()
                .map_or(true, |active| active.is_none());
            if done {
                return;
            }
            notified.await;
        }
    }
}

struct ActiveGuard<S> {
    state: Arc<HttpState<S>>,
}

impl<S> Drop for ActiveGuard<S> {
    fn drop(&mut self) {
        if let Ok(mut active) = self.state.active.lock() {
            *active = None;
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
    let deadline = tokio::time::Instant::now() + state.deadline;
    let body = match body(request).await {
        Ok(body) => body,
        Err(error) => return body_error(&error),
    };
    let request = match serde_json::from_slice::<ChatCompletionRequest>(&body) {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    if request.model != state.model {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    if request.params.stream {
        return stream_chat_response(state, request, deadline).await;
    }
    let permit = match tokio::time::timeout_at(deadline, admit(&state)).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(response)) => return *response,
        Err(_) => return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
    };
    let model = request.model.clone();
    let control = GenerationControl::new(state.write_stall_timeout);
    let active = match activate(&state, &control) {
        Ok(active) => active,
        Err(response) => return *response,
    };
    let (sender, receiver) = oneshot::channel();
    let task_control = control.clone();
    tokio::spawn(async move {
        let _active = active;
        let _permit = permit;
        let mut service = state.service.lock().await;
        let generation = service.complete(request, task_control.clone());
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
    let deadline = tokio::time::Instant::now() + state.deadline;
    let body = match body(request).await {
        Ok(body) => body,
        Err(error) => return body_error(&error),
    };
    let request = match serde_json::from_slice::<CompletionRequest>(&body) {
        Ok(request) => request,
        Err(error) => return api_error(StatusCode::BAD_REQUEST, &error.to_string()),
    };
    if request.model != state.model {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    if request.params.stream {
        return stream_text_response(state, request, deadline).await;
    }
    let permit = match tokio::time::timeout_at(deadline, admit(&state)).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(response)) => return *response,
        Err(_) => return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
    };
    let model = request.model.clone();
    let control = GenerationControl::new(state.write_stall_timeout);
    let active = match activate(&state, &control) {
        Ok(active) => active,
        Err(response) => return *response,
    };
    let (sender, receiver) = oneshot::channel();
    let task_control = control.clone();
    tokio::spawn(async move {
        let _active = active;
        let _permit = permit;
        let mut service = state.service.lock().await;
        let generation = service.complete_text(request, task_control.clone());
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
    let created = unix_time();
    match text_response(&format!("cmpl-{created}"), created, &model, &completion) {
        Ok(body) => Json(body).into_response(),
        Err(error) => generation_error(&error),
    }
}

async fn stream_chat_response<S: ChatService>(
    state: Arc<HttpState<S>>,
    request: ChatCompletionRequest,
    deadline: tokio::time::Instant,
) -> Response {
    if request.model != state.model {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    let permit = match tokio::time::timeout_at(deadline, admit(&state)).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(response)) => return *response,
        Err(_) => return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
    };
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
    let control = GenerationControl::new(state.write_stall_timeout);
    let active = match activate(&state, &control) {
        Ok(active) => active,
        Err(response) => return *response,
    };
    tokio::spawn(async move {
        let _active = active;
        let _permit = permit;
        let mut service = state.service.lock().await;
        let generation = service.stream_chat(request, sender.clone(), control.clone());
        controlled_deadline(deadline, &control, generation, move |result| async move {
            send_stream_error(result, &sender).await;
        })
        .await;
    });
    sse_response(receiver, id, created, model, true)
}

async fn stream_text_response<S: ChatService>(
    state: Arc<HttpState<S>>,
    request: CompletionRequest,
    deadline: tokio::time::Instant,
) -> Response {
    if request.model != state.model {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    let permit = match tokio::time::timeout_at(deadline, admit(&state)).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(response)) => return *response,
        Err(_) => return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
    };
    let model = request.model.clone();
    let created = unix_time();
    let id = format!("cmpl-{created}");
    let (sender, receiver) = mpsc::channel(16);
    let control = GenerationControl::new(state.write_stall_timeout);
    let active = match activate(&state, &control) {
        Ok(active) => active,
        Err(response) => return *response,
    };
    tokio::spawn(async move {
        let _active = active;
        let _permit = permit;
        let mut service = state.service.lock().await;
        let generation = service.stream_text(request, sender.clone(), control.clone());
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
    Json(json!({"object": "list", "data": [model_object(&state.model)]})).into_response()
}

async fn model<S: ChatService>(
    State(state): State<Arc<HttpState<S>>>,
    Path(requested): Path<String>,
) -> Response {
    if requested == state.model {
        Json(model_object(&state.model)).into_response()
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

async fn admit<S>(state: &Arc<HttpState<S>>) -> Result<OwnedSemaphorePermit, Box<Response>> {
    if state.draining.load(Ordering::Acquire) {
        return Err(Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server is draining",
        )));
    }
    if let Ok(permit) = Arc::clone(&state.sequence).try_acquire_owned() {
        return Ok(permit);
    }
    if state
        .queued
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |queued| {
            (queued < state.queue_limit).then_some(queued + 1)
        })
        .is_err()
    {
        return Err(Box::new(api_error(
            StatusCode::TOO_MANY_REQUESTS,
            "request queue is full",
        )));
    }
    let reservation = QueueReservation(&state.queued);
    let permit = Arc::clone(&state.sequence).acquire_owned().await;
    drop(reservation);
    if state.draining.load(Ordering::Acquire) {
        return Err(Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server is draining",
        )));
    }
    permit.map_err(|_| {
        Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server is draining",
        ))
    })
}

fn activate<S>(
    state: &Arc<HttpState<S>>,
    control: &GenerationControl,
) -> Result<ActiveGuard<S>, Box<Response>> {
    let mut active = state.active.lock().map_err(|_| {
        Box::new(api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "admission state failed",
        ))
    })?;
    if state.draining.load(Ordering::Acquire) {
        return Err(Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server is draining",
        )));
    }
    *active = Some(control.clone());
    drop(active);
    Ok(ActiveGuard {
        state: Arc::clone(state),
    })
}

struct QueueReservation<'a>(&'a AtomicU32);

impl Drop for QueueReservation<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
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
    async fn bounds_the_fifo_waiting_queue() {
        let state = state();
        let active = admit(&state).await.unwrap();
        let waiting = admit(&state);
        tokio::pin!(waiting);
        assert!(matches!(
            futures_poll_once(waiting.as_mut()),
            std::task::Poll::Pending
        ));
        assert_eq!(
            admit(&state).await.unwrap_err().status(),
            StatusCode::TOO_MANY_REQUESTS
        );
        drop(active);
        assert!(waiting.await.is_ok());
    }

    #[tokio::test]
    async fn wrong_model_never_enters_admission() {
        let state = state();
        let _active = Arc::clone(&state.sequence).try_acquire_owned().unwrap();
        let chat_request = br#"{"model":"missing","messages":[]}"#;
        let text_request = br#"{"model":"missing","prompt":"hi"}"#;

        let chat_response = chat::<Smoke>(State(Arc::clone(&state)), request(chat_request)).await;
        let text_response = text::<Smoke>(State(Arc::clone(&state)), request(text_request)).await;
        let stream_chat = serde_json::from_slice(chat_request).unwrap();
        let stream_text = serde_json::from_slice(text_request).unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        let stream_chat_response =
            stream_chat_response(Arc::clone(&state), stream_chat, deadline).await;
        let stream_text_response = stream_text_response(state.clone(), stream_text, deadline).await;

        assert_eq!(chat_response.status(), StatusCode::NOT_FOUND);
        assert_eq!(text_response.status(), StatusCode::NOT_FOUND);
        assert_eq!(stream_chat_response.status(), StatusCode::NOT_FOUND);
        assert_eq!(stream_text_response.status(), StatusCode::NOT_FOUND);
        assert_eq!(state.queued.load(Ordering::Acquire), 0);
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
        let state = Arc::new(HttpState {
            model: "smoke".to_owned(),
            service: Mutex::new(DeadlineSmoke {
                release: Arc::clone(&release),
                block_next: true,
            }),
            sequence: Arc::new(Semaphore::new(1)),
            queued: AtomicU32::new(0),
            queue_limit: 1,
            deadline: Duration::from_millis(30),
            write_stall_timeout: Duration::from_secs(1),
            draining: std::sync::atomic::AtomicBool::new(false),
            active: StdMutex::new(None),
            drained: tokio::sync::Notify::new(),
        });
        let body = br#"{"model":"smoke","messages":[]}"#;
        let started = tokio::time::Instant::now();

        let first = chat::<DeadlineSmoke>(State(Arc::clone(&state)), request(body)).await;

        assert_eq!(first.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(started.elapsed() < Duration::from_millis(250));
        assert_eq!(state.sequence.available_permits(), 0);
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
        let permit = admit(&state).await.unwrap();
        let control = GenerationControl::new(Duration::from_secs(1));
        let active = activate(&state, &control).unwrap();
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
        drop(permit);
        drain.await;
        assert_eq!(
            admit(&state).await.unwrap_err().status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
    }

    fn state() -> Arc<HttpState<Smoke>> {
        Arc::new(HttpState {
            model: "smoke".to_owned(),
            service: Mutex::new(Smoke),
            sequence: Arc::new(Semaphore::new(1)),
            queued: AtomicU32::new(0),
            queue_limit: 1,
            deadline: Duration::from_secs(1),
            write_stall_timeout: Duration::from_secs(1),
            draining: std::sync::atomic::AtomicBool::new(false),
            active: StdMutex::new(None),
            drained: tokio::sync::Notify::new(),
        })
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
}
