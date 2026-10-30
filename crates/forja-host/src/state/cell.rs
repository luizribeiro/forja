use super::{StateSequenceHandle, StateStorage, pool_error};
use crate::{Limits, compute, invalid_handle};
use forja_core::{Backend, Tensor};
use std::sync::Arc;

/// One opaque state plane with synchronous publication after each successful update.
pub struct StateCell<B: Backend> {
    storage: Arc<StateStorage<B>>,
    sequence: StateSequenceHandle<B>,
    serial: u64,
    submitted: bool,
    profiled: bool,
    profile_scope: Option<Arc<forja_core::ProfileScope>>,
    profile: Option<forja_core::SubmissionProfile>,
}

impl<B: Backend> StateCell<B> {
    /// Reserves a zeroed sequence from a single-plane pool.
    ///
    /// # Errors
    /// Refuses multiple planes or unavailable storage.
    pub fn new(storage: Arc<StateStorage<B>>) -> Result<Self, compute::Error> {
        if storage.pool.layout().planes().len() != 1 {
            return Err(invalid_handle("a state cell requires exactly one plane"));
        }
        let sequence = storage.reserve()?;
        Ok(Self {
            storage,
            sequence,
            serial: 0,
            submitted: false,
            profiled: false,
            profile_scope: None,
            profile: None,
        })
    }

    /// Executes one authenticated delta-rule update and publishes on completion.
    ///
    /// # Errors
    /// Returns validation, quota or execution failures without publishing failed work.
    pub fn delta_rule(
        &mut self,
        inputs: [&Tensor; 5],
        output: &Tensor,
        limits: Limits,
    ) -> Result<Option<std::time::Duration>, compute::Error> {
        self.update(limits, |recording| recording.delta_rule(0, inputs, output))
    }

    /// Executes one authenticated convolution update and publishes on completion.
    ///
    /// # Errors
    /// Returns validation, quota or execution failures without publishing failed work.
    pub fn causal_convolution(
        &mut self,
        input: &Tensor,
        weights: &Tensor,
        output: &Tensor,
        limits: Limits,
    ) -> Result<Option<std::time::Duration>, compute::Error> {
        self.update(limits, |recording| {
            recording.causal_convolution(0, input, weights, output)
        })
    }

    pub(super) const fn was_submitted(&self) -> bool {
        self.submitted
    }

    pub(super) fn set_profile_context(
        &mut self,
        enabled: bool,
        scope: Option<Arc<forja_core::ProfileScope>>,
    ) {
        self.profiled = enabled;
        self.profile_scope = scope;
    }

    pub(super) fn take_profile(&mut self) -> Option<forja_core::SubmissionProfile> {
        self.profile.take()
    }

    pub(super) fn update(
        &mut self,
        limits: Limits,
        record: impl FnOnce(&mut super::StateRecording<B>) -> Result<(), compute::Error>,
    ) -> Result<Option<std::time::Duration>, compute::Error> {
        self.submitted = false;
        self.profile = None;
        let serial = self
            .serial
            .checked_add(1)
            .ok_or_else(|| invalid_handle("state serial exhausted"))?;
        let transaction = self.sequence.begin(serial).map_err(pool_error)?;
        self.serial = serial;
        let mut recording =
            match self
                .storage
                .record_with_limits(&self.sequence, transaction.clone(), limits)
            {
                Ok(recording) => recording,
                Err(error) => {
                    self.sequence.cancel(&transaction).map_err(pool_error)?;
                    return Err(error);
                }
            };
        recording
            .commands
            .set_profile_scope(self.profile_scope.clone());
        record(&mut recording)?;
        self.submitted = true;
        let (time, profile) = recording
            .submit_with_profiling(self.profiled)?
            .finish_with_profile()?;
        self.profile = profile;
        Ok(time)
    }
}
