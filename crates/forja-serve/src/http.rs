use std::{future::Future, pin::Pin, sync::Arc, time::UNIX_EPOCH};

use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::post,
};
use forja_core::Backend;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::engine::ErrorKind;
use crate::{ChatCompletionRequest, Completion, GenerationError, ServingEngine, chat_response};

/// Asynchronous chat generation used by the HTTP transport.
pub trait ChatService: Send + 'static {
    /// Served model identifier.
    fn model(&self) -> &str;

    /// Completes one chat request.
    fn complete(
        &mut self,
        request: ChatCompletionRequest,
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
}

/// Builds the OpenAI-compatible chat-completions router.
pub fn chat_router<S: ChatService>(service: S) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat::<S>))
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
    let created = std::time::SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |duration| duration.as_secs());
    match chat_response(&format!("chatcmpl-{created}"), created, &model, &completion) {
        Ok(body) => Json(body).into_response(),
        Err(error) => generation_error(&error),
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
}
