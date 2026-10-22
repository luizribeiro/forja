use std::{
    sync::{Arc, Mutex as StdMutex},
    time::Instant,
};

use forja_core::Backend;
use tokio::sync::{mpsc, oneshot};

use super::ServingEngine;
use crate::{
    ChatCompletionRequest, Completion, CompletionRequest, GenerationControl, GenerationError,
    GenerationEvent, ServingMetrics, http::ServiceCommand, metrics::observe,
};

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

    async fn run_serial<B>(self, engine: &mut ServingEngine<B>)
    where
        B: Backend + Send + Sync + 'static,
    {
        match self {
            Self::Chat {
                request,
                control,
                reply,
                ..
            } => {
                drop(reply.send(engine.complete_chat_controlled(&request, &control).await));
            }
            Self::Text {
                request,
                control,
                reply,
                ..
            } => {
                drop(reply.send(engine.complete_text_controlled(&request, &control).await));
            }
            Self::StreamChat {
                request,
                events,
                control,
                reply,
                ..
            } => {
                let result = engine
                    .stream_chat_controlled(&request, &events, &control)
                    .await;
                drop(reply.send(result));
            }
            Self::StreamText {
                request,
                events,
                control,
                reply,
                ..
            } => {
                let result = engine
                    .stream_text_controlled(&request, &events, &control)
                    .await;
                drop(reply.send(result));
            }
        }
    }
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
        while let Some(command) = receiver.recv().await {
            let envelope = RequestEnvelope::from_command(command);
            let queued = envelope.queued().elapsed();
            let scheduled = Instant::now();
            envelope.run_serial(&mut self).await;
            if let Ok(mut metrics) = shared_metrics.lock() {
                self.metrics.queue_time = metrics.queue_time;
                self.metrics.scheduler_time = metrics.scheduler_time;
                observe(&mut self.metrics.queue_time, queued);
                observe(&mut self.metrics.scheduler_time, scheduled.elapsed());
                *metrics = self.metrics.clone();
            }
        }
    }
}
