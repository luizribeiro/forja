use std::{
    sync::{Arc, Mutex as StdMutex},
    time::Instant,
};

use forja_core::Backend;
use tokio::sync::mpsc;

use super::ServingEngine;
use crate::{ServingMetrics, http::ServiceCommand, metrics::observe};

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
            let queued = command.queued().elapsed();
            let scheduled = Instant::now();
            self.run_command(command).await;
            if let Ok(mut metrics) = shared_metrics.lock() {
                self.metrics.queue_time = metrics.queue_time;
                self.metrics.scheduler_time = metrics.scheduler_time;
                observe(&mut self.metrics.queue_time, queued);
                observe(&mut self.metrics.scheduler_time, scheduled.elapsed());
                *metrics = self.metrics.clone();
            }
        }
    }

    async fn run_command(&mut self, command: ServiceCommand) {
        match command {
            ServiceCommand::Chat {
                request,
                control,
                reply,
                ..
            } => {
                let result = self.complete_chat_controlled(&request, &control).await;
                drop(reply.send(result));
            }
            ServiceCommand::Text {
                request,
                control,
                reply,
                ..
            } => {
                let result = self.complete_text_controlled(&request, &control).await;
                drop(reply.send(result));
            }
            ServiceCommand::StreamChat {
                request,
                events,
                control,
                reply,
                ..
            } => {
                let result = self
                    .stream_chat_controlled(&request, &events, &control)
                    .await;
                drop(reply.send(result));
            }
            ServiceCommand::StreamText {
                request,
                events,
                control,
                reply,
                ..
            } => {
                let result = self
                    .stream_text_controlled(&request, &events, &control)
                    .await;
                drop(reply.send(result));
            }
        }
    }
}
