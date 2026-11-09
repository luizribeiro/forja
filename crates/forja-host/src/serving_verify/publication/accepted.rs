use super::{Backend, EngineBatch, Host, VerifyPublication, compute, invalid_handle};
use crate::{EngineBatchOutput, EngineRunner, speculative::VerifyDecision};

impl<B: Backend + Send + Sync + 'static> EngineRunner<B> {
    pub(crate) fn prepare_verify_publication(
        &mut self,
        pool: u32,
        slot: u32,
        input: &EngineBatch,
        checkpoints: &mut [Option<(u32, forja_core::SequenceCheckpoint)>],
    ) -> Result<Option<VerifyPublication>, compute::Error> {
        (|| {
            self.store.data_mut().advance_paged_batch_bank(slot)?;
            self.store.data().verify_publication(pool, input)
        })()
        .or_else(|error| {
            self.rollback_batch_appends(pool, checkpoints)?;
            Err(error)
        })
    }

    /// Publishes the accepted verify row and releases the batch's output handles.
    ///
    /// Read and scan the returned target samples before calling this method.
    /// A publication error requires restoring the lane's pre-round checkpoint.
    ///
    /// # Errors
    /// Refuses foreign outputs, ordinary batches, incomplete work or invalid cuts.
    pub fn publish_verified_batch(
        &mut self,
        mut output: EngineBatchOutput,
        decision: &VerifyDecision<'_>,
    ) -> Result<(), compute::Error> {
        if output.tokens.runner_id != self.id || output.logits.runner_id != self.id {
            return Err(invalid_handle("engine output belongs to another runner"));
        }
        let result = output
            .verify
            .take()
            .ok_or_else(|| invalid_handle("batch is not awaiting verify acceptance"))
            .and_then(|publication| {
                publication.publish(self.store.data_mut(), decision.accepted())
            });
        let release = self.release_batch_output(output);
        result.and(release)
    }
}

impl VerifyPublication {
    pub(super) fn publish<B: Backend + 'static>(
        &self,
        host: &mut Host<B>,
        accepted: u32,
    ) -> Result<(), compute::Error> {
        if accepted >= self.rows {
            return Err(invalid_handle("accepted verify row exceeds the batch"));
        }
        if let Some(transaction) = &self.state {
            host.lane_state(self.pool, self.lane)?
                .ok_or_else(|| invalid_handle("verify state lane is unavailable"))?
                .publish_accepted(transaction, accepted)
                .map_err(invalid_handle)?;
            self.invalidate_after_cut(host, accepted)
        } else {
            self.publish_kv(host, accepted)
        }
    }
}
