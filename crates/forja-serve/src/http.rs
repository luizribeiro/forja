use std::{future::Future, pin::Pin, sync::Arc, time::UNIX_EPOCH};

use axum::{
    Json, Router,
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
};
use forja_core::Backend;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::engine::ErrorKind;
use crate::{
    ChatCompletionRequest, Completion, CompletionRequest, GenerationError, ServingEngine,
    chat_response, text_response,
};

/// Asynchronous chat generation used by the HTTP transport.
pub trait ChatService: Send + 'static {
    /// Served model identifier.
    fn model(&self) -> &str;

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
    Router::new()
        .route("/v1/chat/completions", post(chat::<S>))
        .route("/v1/completions", post(text::<S>))
        .route("/v1/models", get(models::<S>))
        .route("/v1/models/{model}", get(model::<S>))
        .with_state(Arc::new(Mutex::new(service)))
}

async fn chat<S: ChatService>(State(service): State<Arc<Mutex<S>>>, body: Bytes) -> Response {
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
    let mut service = service.lock().await;
    if request.model != service.model() {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    let model = request.model.clone();
    let completion = match service.complete(request).await {
        Ok(completion) => completion,
        Err(error) => return generation_error(&error),
    };
    let created = unix_time();
    match chat_response(&format!("chatcmpl-{created}"), created, &model, &completion) {
        Ok(body) => Json(body).into_response(),
        Err(error) => generation_error(&error),
    }
}

async fn text<S: ChatService>(State(service): State<Arc<Mutex<S>>>, body: Bytes) -> Response {
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
    let mut service = service.lock().await;
    if request.model != service.model() {
        return api_error(StatusCode::NOT_FOUND, "model is not loaded");
    }
    let model = request.model.clone();
    let completion = match service.complete_text(request).await {
        Ok(completion) => completion,
        Err(error) => return generation_error(&error),
    };
    let created = unix_time();
    match text_response(&format!("cmpl-{created}"), created, &model, &completion) {
        Ok(body) => Json(body).into_response(),
        Err(error) => generation_error(&error),
    }
}

async fn models<S: ChatService>(State(service): State<Arc<Mutex<S>>>) -> Response {
    let model = service.lock().await.model().to_owned();
    Json(json!({"object": "list", "data": [model_object(&model)]})).into_response()
}

async fn model<S: ChatService>(
    State(service): State<Arc<Mutex<S>>>,
    Path(requested): Path<String>,
) -> Response {
    let model = service.lock().await.model().to_owned();
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
        let response = chat::<Smoke>(
            State(Arc::new(Mutex::new(Smoke))),
            Bytes::copy_from_slice(body),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn serves_text_completions_and_model_metadata() {
        let state = Arc::new(Mutex::new(Smoke));
        let response = text::<Smoke>(
            State(Arc::clone(&state)),
            Bytes::from_static(br#"{"model":"smoke","prompt":"hi"}"#),
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
}
