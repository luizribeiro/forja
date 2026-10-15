use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicU32, Ordering},
    },
    time::{Duration, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, FromRequest, Path, Request, State, rejection::BytesRejection},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use forja_core::Backend;
use serde_json::{Value, json};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};

use crate::engine::ErrorKind;
use crate::{
    ChatCompletionRequest, Completion, CompletionRequest, GenerationError, ServingEngine,
    chat_response, text_response,
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

    /// Completes one chat request.
    fn complete(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>;

    /// Completes one unformatted text request.
    fn complete_text(
        &mut self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>;
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

    fn complete(
        &mut self,
        request: ChatCompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>> {
        Box::pin(async move { self.complete_chat(&request).await })
    }

    fn complete_text(
        &mut self,
        request: CompletionRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>> {
        Box::pin(async move { self.complete_text(&request).await })
    }
}

/// Builds the OpenAI-compatible chat-completions router.
pub fn chat_router<S: ChatService>(service: S) -> Router {
    let max_body_bytes = service.max_body_bytes();
    let state = Arc::new(HttpState {
        queue_limit: service.queue_length(),
        deadline: service.request_deadline(),
        service: Mutex::new(service),
        sequence: Arc::new(Semaphore::new(1)),
        queued: AtomicU32::new(0),
    });
    Router::new()
        .route("/v1/chat/completions", post(chat::<S>))
        .route("/v1/completions", post(text::<S>))
        .route("/v1/models", get(models::<S>))
        .route("/v1/models/{model}", get(model::<S>))
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(state)
}

struct HttpState<S> {
    service: Mutex<S>,
    sequence: Arc<Semaphore>,
    queued: AtomicU32,
    queue_limit: u32,
    deadline: Duration,
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
    if request.params.stream {
        return api_error(
            StatusCode::BAD_REQUEST,
            "streaming is not available on this route",
        );
    }
    let _permit = match tokio::time::timeout_at(deadline, admit(&state)).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(response)) => return *response,
        Err(_) => return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
    };
    let mut service = state.service.lock().await;
    if request.model != service.model() {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    let model = request.model.clone();
    let completion = match tokio::time::timeout_at(deadline, service.complete(request)).await {
        Ok(Ok(completion)) => completion,
        Ok(Err(error)) => return generation_error(&error),
        Err(_) => return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
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
    if request.params.stream {
        return api_error(
            StatusCode::BAD_REQUEST,
            "streaming is not available on this route",
        );
    }
    let _permit = match tokio::time::timeout_at(deadline, admit(&state)).await {
        Ok(Ok(permit)) => permit,
        Ok(Err(response)) => return *response,
        Err(_) => return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
    };
    let mut service = state.service.lock().await;
    if request.model != service.model() {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    let model = request.model.clone();
    let completion = match tokio::time::timeout_at(deadline, service.complete_text(request)).await {
        Ok(Ok(completion)) => completion,
        Ok(Err(error)) => return generation_error(&error),
        Err(_) => return api_error(StatusCode::GATEWAY_TIMEOUT, "request deadline exceeded"),
    };
    let created = unix_time();
    match text_response(&format!("cmpl-{created}"), created, &model, &completion) {
        Ok(body) => Json(body).into_response(),
        Err(error) => generation_error(&error),
    }
}

async fn models<S: ChatService>(State(state): State<Arc<HttpState<S>>>) -> Response {
    let model = state.service.lock().await.model().to_owned();
    Json(json!({"object": "list", "data": [model_object(&model)]})).into_response()
}

async fn model<S: ChatService>(
    State(state): State<Arc<HttpState<S>>>,
    Path(requested): Path<String>,
) -> Response {
    let model = state.service.lock().await.model().to_owned();
    if requested == model {
        Json(model_object(&model)).into_response()
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
    permit.map_err(|_| {
        Box::new(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "server is draining",
        ))
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
    (
        status,
        Json(json!({"error": {
            "message": message,
            "type": error_type,
            "param": Value::Null,
            "code": Value::Null,
        }})),
    )
        .into_response()
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

        fn complete(
            &mut self,
            _: ChatCompletionRequest,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            Box::pin(async {
                Ok(Completion {
                    text: "forged".to_owned(),
                    finish_reason: FinishReason::Stop,
                    usage: Usage {
                        prompt_tokens: 4,
                        completion_tokens: 1,
                    },
                })
            })
        }

        fn complete_text(
            &mut self,
            _: CompletionRequest,
        ) -> Pin<Box<dyn Future<Output = Result<Completion, GenerationError>> + Send + '_>>
        {
            self.complete(ChatCompletionRequest {
                model: "smoke".to_owned(),
                messages: Vec::new(),
                params: crate::CompletionParams::default(),
            })
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

    fn state() -> Arc<HttpState<Smoke>> {
        Arc::new(HttpState {
            service: Mutex::new(Smoke),
            sequence: Arc::new(Semaphore::new(1)),
            queued: AtomicU32::new(0),
            queue_limit: 1,
            deadline: Duration::from_secs(1),
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
