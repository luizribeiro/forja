use std::{sync::Arc, time::Duration};

use crate::http::{AdmissionPolicy, ServiceActor};
use crate::{
    ChatService, Completion, CompletionRequest, GenerationControl, GenerationError, ServingMetrics,
};

/// In-process access to the same scheduler used by serving transports.
pub struct CompletionSession<S: ChatService> {
    actor: ServiceActor<S>,
    policy: AdmissionPolicy,
    stall: Duration,
    finalizer: tokio::task::JoinHandle<()>,
}

impl<S: ChatService> CompletionSession<S> {
    /// Starts the service's bounded scheduler without a network listener.
    pub fn new(service: S) -> Self {
        let policy = AdmissionPolicy {
            config: Arc::new(service.host_config()),
        };
        let stall = service.write_stall_timeout();
        let capacity = usize::try_from(service.queue_length())
            .unwrap_or(usize::MAX)
            .max(1);
        let (actor, finalizer) = ServiceActor::spawn_tracked(service, capacity);
        Self {
            actor,
            policy,
            stall,
            finalizer,
        }
    }

    /// Admits a group of text requests before awaiting any result.
    ///
    /// # Errors
    /// Returns admission, generation, or scheduler failures.
    pub async fn complete_texts(
        &self,
        requests: Vec<CompletionRequest>,
    ) -> Result<Vec<Completion>, GenerationError> {
        let mut pending = Vec::with_capacity(requests.len());
        for request in requests {
            pending.push(
                self.actor
                    .complete_text_with_policy(
                        self.policy.clone(),
                        request,
                        GenerationControl::new(self.stall),
                    )
                    .map_err(|_| GenerationError::unavailable("completion mailbox unavailable"))?,
            );
        }
        let mut completed = Vec::with_capacity(pending.len());
        for reply in pending {
            completed.push(
                reply
                    .await
                    .map_err(|_| GenerationError::unavailable("completion scheduler stopped"))??,
            );
        }
        self.checkpoint().await?;
        Ok(completed)
    }

    /// Drains submitted work and returns a current observation boundary.
    ///
    /// # Errors
    /// Returns a scheduler or backend failure.
    pub async fn checkpoint(&self) -> Result<ServingMetrics, GenerationError> {
        self.actor.quiesce().await?;
        Ok(self.actor.metrics())
    }

    /// Returns the scheduler's cumulative observations after completed work.
    #[must_use]
    pub fn metrics(&self) -> ServingMetrics {
        self.actor.metrics()
    }

    /// Drains the scheduler and waits for its runtime to be released.
    ///
    /// # Errors
    /// Returns a scheduler shutdown failure.
    pub async fn close(self) -> Result<(), GenerationError> {
        self.actor.quiesce().await?;
        drop(self.actor);
        self.finalizer
            .await
            .map_err(|_| GenerationError::unavailable("completion scheduler failed"))
    }
}
