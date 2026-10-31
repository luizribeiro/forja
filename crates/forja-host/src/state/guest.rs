use super::StateCell;
use crate::{
    GpuReservation, Host, ImportKind, Limits, SubmissionAccess, TensorEntry, compute, core_dtype,
    duration_ns, guest_error, invalid_handle, saturating_add, saturating_increment,
};
use forja_core::{Backend, BackendError, StateLayout, StatePlane};
use std::{
    any::Any,
    future::Future,
    sync::Arc,
    time::{Duration, Instant},
};
use wasmtime::component::Resource;

/// Opaque component resource; the guest can never obtain its plane tensors.
pub struct StateEntry {
    cell: Box<dyn Any + Send + Sync>,
}

impl<B: Backend + Send + Sync + 'static> Host<B> {
    fn state_tensors(
        &self,
        resources: &[Resource<TensorEntry>],
    ) -> Result<Vec<TensorEntry>, compute::Error> {
        resources
            .iter()
            .map(|resource| {
                let entry = self.entry(resource)?;
                if entry.symbolic.is_some() {
                    return Err(invalid_handle(
                        "eager state updates require fixed tensor views",
                    ));
                }
                entry
                    .buffer
                    .taints
                    .wait_for_writes(entry.tensor.buffer(), self.limits.submission_timeout)
                    .map_err(guest_error)?;
                Ok(entry.clone())
            })
            .collect()
    }

    fn update_state(
        &mut self,
        resource: &Resource<StateEntry>,
        inputs: &[TensorEntry],
        output: &TensorEntry,
        run: impl FnOnce(&mut StateCell<B>, Limits) -> Result<Option<Duration>, compute::Error>,
    ) -> Result<(), compute::Error> {
        let _timer = self.import_timer(ImportKind::Submit);
        let limits = self.limits;
        let reservation = GpuReservation::new(
            Arc::clone(&self.gpu_time_ns),
            duration_ns(limits.gpu_time_budget),
            duration_ns(limits.submission_timeout).max(1),
        )
        .map_err(guest_error)?;
        let taints = Arc::clone(&self.taints);
        let profile = self.active_profile.clone();
        let scope = self.current_profile_scope.clone();
        let cell = self.state_cell(resource)?;
        cell.set_profile_context(profile.is_some(), scope);
        let mut access = SubmissionAccess::default();
        access.record(inputs, std::slice::from_ref(output));
        let order = taints
            .queue
            .lock()
            .map_err(|_| guest_error(BackendError::ExecutionFailed))?;
        let flight = taints.begin(&access).map_err(guest_error)?;
        let started = Instant::now();
        let result = run(cell, limits);
        drop(order);
        let submitted = cell.was_submitted();
        crate::record_submission_profile(profile.as_ref(), cell.take_profile());
        let outcome = flight.finish(
            result
                .as_ref()
                .map(|_| ())
                .map_err(|_| BackendError::ExecutionFailed),
            submitted,
            limits.submission_timeout,
        );
        let charged = match result {
            Ok(time) => time
                .map_or_else(|| duration_ns(started.elapsed()), duration_ns)
                .max(1),
            Err(_) if submitted => reservation.amount(),
            Err(_) => 0,
        };
        let accounting = reservation.settle(charged).map_err(guest_error);
        let time = result?;
        accounting?;
        outcome.map_err(guest_error)?;
        saturating_increment(&self.queued_submissions);
        saturating_increment(&self.completed_submissions);
        if let Some(time) = time {
            saturating_increment(&self.timed_submissions);
            saturating_add(&self.completed_gpu_time_ns, duration_ns(time));
        }
        self.submission_notify.notify_waiters();
        Ok(())
    }

    fn state_cell(
        &mut self,
        resource: &Resource<StateEntry>,
    ) -> Result<&mut StateCell<B>, compute::Error> {
        self.table
            .get_mut(resource)
            .map_err(invalid_handle)?
            .cell
            .downcast_mut()
            .ok_or_else(|| invalid_handle("state backend mismatch"))
    }

    fn update_delta_state(
        &mut self,
        resource: &Resource<StateEntry>,
        inputs: &[Resource<TensorEntry>],
        output: Resource<TensorEntry>,
        variant: Option<&str>,
    ) -> Result<(), compute::Error> {
        let inputs = self.state_tensors(inputs)?;
        let [q, k, v, decay, beta] = inputs.as_slice() else {
            return Err(invalid_handle("delta rule requires five ordinary inputs"));
        };
        let output = self.state_tensors(&[output])?;
        self.update_state(resource, &inputs, &output[0], |cell, limits| {
            let operands = [&q.tensor, &k.tensor, &v.tensor, &decay.tensor, &beta.tensor];
            if let Some(variant) = variant {
                cell.delta_rule_block(operands, &output[0].tensor, limits, Some(variant))
            } else {
                cell.delta_rule(operands, &output[0].tensor, limits)
            }
        })
    }
}

impl<B: Backend + Send + Sync + 'static> compute::HostStatePlane for Host<B> {
    fn create(
        &mut self,
        dtype: compute::Dtype,
        shape: Vec<u32>,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<StateEntry>, compute::Error>>> + Send
    {
        let result = (|| {
            self.check_handle_quota()?;
            let plane = StatePlane::new(core_dtype(dtype), shape).map_err(invalid_handle)?;
            let layout = StateLayout::new(vec![plane]).map_err(invalid_handle)?;
            let cell = StateCell::new(self.create_state_storage(layout, 1)?)?;
            let resource = self
                .table
                .push(StateEntry {
                    cell: Box::new(cell),
                })
                .map_err(invalid_handle)?;
            self.live_handles += 1;
            Ok(resource)
        })();
        std::future::ready(Ok(result))
    }

    fn delta_rule(
        &mut self,
        resource: Resource<StateEntry>,
        inputs: Vec<Resource<TensorEntry>>,
        output: Resource<TensorEntry>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let result = self.update_delta_state(&resource, &inputs, output, None);
        std::future::ready(Ok(result))
    }

    fn delta_rule_block(
        &mut self,
        resource: Resource<StateEntry>,
        inputs: Vec<Resource<TensorEntry>>,
        output: Resource<TensorEntry>,
        variant: String,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let result = self.update_delta_state(&resource, &inputs, output, Some(&variant));
        std::future::ready(Ok(result))
    }

    fn causal_convolution(
        &mut self,
        resource: Resource<StateEntry>,
        input: Resource<TensorEntry>,
        weights: Resource<TensorEntry>,
        output: Resource<TensorEntry>,
    ) -> impl Future<Output = wasmtime::Result<Result<(), compute::Error>>> + Send {
        let result = (|| {
            let tensors = self.state_tensors(&[input, weights, output])?;
            self.update_state(&resource, &tensors[..2], &tensors[2], |cell, limits| {
                cell.causal_convolution(
                    &tensors[0].tensor,
                    &tensors[1].tensor,
                    &tensors[2].tensor,
                    limits,
                )
            })
        })();
        std::future::ready(Ok(result))
    }

    fn drop(
        &mut self,
        resource: Resource<StateEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        let result = self
            .table
            .delete(resource)
            .map(|_| {
                self.live_handles -= 1;
            })
            .map_err(wasmtime::Error::msg);
        std::future::ready(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use compute::HostStatePlane;
    #[tokio::test]
    async fn resources_charge_quota_and_reject_invalid_updates() {
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 3, 4096),
        );
        let before = host.live_bytes.load(std::sync::atomic::Ordering::Acquire);
        let state = host
            .create(compute::Dtype::F32, vec![1, 1])
            .await
            .unwrap()
            .unwrap();
        let output = host.alloc(compute::Dtype::F32, &[1, 1]).unwrap();
        let borrow = || Resource::new_borrow(state.rep());
        let result = host
            .delta_rule(borrow(), vec![], Resource::new_borrow(output.rep()))
            .await
            .unwrap();
        assert!(result.is_err());
        assert_eq!(host.live_handles, 2);
        host.drop_tensor(output).unwrap();
        HostStatePlane::drop(&mut host, state).await.unwrap();
        assert_eq!(host.live_handles, 0);
        assert_eq!(
            host.live_bytes.load(std::sync::atomic::Ordering::Acquire),
            before
        );
        assert!(
            host.create(compute::Dtype::U32, vec![1])
                .await
                .unwrap()
                .is_err()
        );
    }

    #[tokio::test]
    async fn delta_block_import_publishes_with_current_limits() {
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(1_048_576, 8, 1_048_576, 128, 1_048_576),
        );
        let state = host
            .create(compute::Dtype::F32, vec![1, 1, 1])
            .await
            .unwrap()
            .unwrap();
        let mut inputs = Vec::new();
        for shape in [
            vec![2, 1, 1],
            vec![2, 1, 1],
            vec![2, 1, 1],
            vec![2, 1],
            vec![2, 1],
        ] {
            inputs.push(host.alloc(compute::Dtype::F32, &shape).unwrap());
        }
        let output = host.alloc(compute::Dtype::F32, &[2, 1, 1]).unwrap();
        let before = host.live_bytes.load(std::sync::atomic::Ordering::Acquire);
        let invoke_inputs = || {
            inputs
                .iter()
                .map(|input| Resource::new_borrow(input.rep()))
                .collect()
        };
        host.limits.work_per_dispatch = 0;
        assert!(
            host.delta_rule_block(
                Resource::new_borrow(state.rep()),
                invoke_inputs(),
                Resource::new_borrow(output.rep()),
                "delta-rule-block.wy64-fixed".to_owned()
            )
            .await
            .unwrap()
            .is_err()
        );
        assert_eq!(
            host.live_bytes.load(std::sync::atomic::Ordering::Acquire),
            before
        );
        host.limits.work_per_dispatch = 1_048_576;
        host.delta_rule_block(
            Resource::new_borrow(state.rep()),
            invoke_inputs(),
            Resource::new_borrow(output.rep()),
            "delta-rule-block.wy64-fixed".to_owned(),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(
            host.completed_submissions
                .load(std::sync::atomic::Ordering::Acquire),
            1
        );
        assert_eq!(
            host.live_bytes.load(std::sync::atomic::Ordering::Acquire),
            before
        );
    }
    #[tokio::test]
    async fn execution_errors_take_precedence_over_settlement_errors() {
        use std::sync::atomic::Ordering;
        for execution_failed in [false, true] {
            let mut host = Host::new(
                forja_cpu::CpuBackend::new(),
                Limits::new(65536, 8, 4096, 8, 4096),
            );
            let state = host
                .create(compute::Dtype::F32, vec![1, 1])
                .await
                .unwrap()
                .unwrap();
            let y = host.alloc(compute::Dtype::F32, &[1, 1]).unwrap();
            let output = host.entry(&y).unwrap().clone();
            let total = Arc::clone(&host.gpu_time_ns);
            let error = host
                .update_state(&state, &[], &output, |_, _| {
                    total.store(0, Ordering::Release);
                    if execution_failed {
                        Err(invalid_handle("state execution failed"))
                    } else {
                        Ok(None)
                    }
                })
                .unwrap_err();
            if execution_failed {
                assert!(matches!(error, compute::Error::InvalidHandle(message)
                    if message == "state execution failed"));
            } else {
                assert!(matches!(error, compute::Error::BackendExecution(_)));
            }
            assert_eq!(host.completed_submissions.load(Ordering::Acquire), 0);
        }
    }
    async fn convolve<B: Backend + Send + Sync + 'static>(
        host: &mut Host<B>,
        handles: [u32; 4],
    ) -> Result<(), compute::Error> {
        host.causal_convolution(
            Resource::new_borrow(handles[0]),
            Resource::new_borrow(handles[1]),
            Resource::new_borrow(handles[2]),
            Resource::new_borrow(handles[3]),
        )
        .await
        .unwrap()
    }
    #[tokio::test]
    async fn state_updates_use_current_admission_limits_without_advancing_on_refusal() {
        use std::sync::atomic::Ordering;
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 8, 4096),
        );
        let state = host
            .create(compute::Dtype::F32, vec![1, 1])
            .await
            .unwrap()
            .unwrap();
        let x = host.alloc(compute::Dtype::F32, &[1, 1]).unwrap();
        let weights = host.alloc(compute::Dtype::F32, &[2, 1]).unwrap();
        let y = host.alloc(compute::Dtype::F32, &[1, 1]).unwrap();
        host.write(&x, &2_f32.to_le_bytes()).unwrap();
        host.write(
            &weights,
            &[1_f32.to_le_bytes(), 1_f32.to_le_bytes()].concat(),
        )
        .unwrap();
        let handles = [state.rep(), x.rep(), weights.rep(), y.rep()];
        host.limits.gpu_time_budget = Duration::ZERO;
        assert!(matches!(
            convolve(&mut host, handles).await,
            Err(compute::Error::Quota(_))
        ));
        host.limits.gpu_time_budget = Duration::MAX;
        host.limits.work_per_dispatch = 0;
        assert!(matches!(
            convolve(&mut host, handles).await,
            Err(compute::Error::Quota(_))
        ));
        host.limits.work_per_dispatch = u64::MAX;
        host.limits.live_bytes = host.live_bytes.load(Ordering::Acquire);
        assert!(matches!(
            convolve(&mut host, handles).await,
            Err(compute::Error::Quota(_))
        ));
        host.limits.live_bytes = 65536;
        convolve(&mut host, handles).await.unwrap();
        assert_eq!(
            host.backend.read(&host.entry(&y).unwrap().tensor).unwrap(),
            2_f32.to_le_bytes()
        );
        assert_eq!(host.queued_submissions.load(Ordering::Acquire), 1);
        assert_eq!(host.completed_submissions.load(Ordering::Acquire), 1);
        assert!(host.gpu_time_ns.load(Ordering::Acquire) > 0);
    }
    #[tokio::test]
    async fn validation_refusals_refund_gpu_budget_before_a_valid_update() {
        use std::sync::atomic::Ordering;
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 8, 4096)
                .with_gpu_limits(Duration::from_secs(1), Duration::from_secs(1)),
        );
        let state = host
            .create(compute::Dtype::F32, vec![1, 1])
            .await
            .unwrap()
            .unwrap();
        let x = host.alloc(compute::Dtype::F32, &[1, 1]).unwrap();
        let weights = host.alloc(compute::Dtype::F32, &[2, 1]).unwrap();
        let y = host.alloc(compute::Dtype::F32, &[1, 1]).unwrap();
        let handles = [state.rep(), x.rep(), weights.rep(), y.rep()];
        host.limits.work_per_dispatch = 0;
        for _ in 0..7 {
            assert!(matches!(
                convolve(&mut host, handles).await,
                Err(compute::Error::Quota(_))
            ));
            assert_eq!(host.gpu_time_ns.load(Ordering::Acquire), 0);
        }
        host.limits.work_per_dispatch = u64::MAX;
        convolve(&mut host, handles).await.unwrap();
        assert_eq!(host.completed_submissions.load(Ordering::Acquire), 1);
    }
    #[tokio::test]
    async fn failed_state_outputs_are_tainted_and_tainted_inputs_are_refused() {
        use std::sync::atomic::Ordering;
        for input_tainted in [false, true] {
            let (control, mut host) = super::super::lifecycle_tests::controlled_host();
            let state = host
                .create(compute::Dtype::F32, vec![1, 1])
                .await
                .unwrap()
                .unwrap();
            let x = host.alloc(compute::Dtype::F32, &[1, 1]).unwrap();
            let weights = host.alloc(compute::Dtype::F32, &[2, 1]).unwrap();
            let y = host.alloc(compute::Dtype::F32, &[1, 1]).unwrap();
            if input_tainted {
                let id = host.entry(&x).unwrap().tensor.buffer();
                host.taints
                    .state
                    .lock()
                    .unwrap()
                    .tainted
                    .insert(id, (0, BackendError::ExecutionFailed));
            } else {
                control.failure.store(2, Ordering::Release);
            }
            assert!(
                convolve(&mut host, [state.rep(), x.rep(), weights.rep(), y.rep()])
                    .await
                    .is_err()
            );
            if !input_tainted {
                assert!(
                    host.state_tensors(&[Resource::new_borrow(y.rep())])
                        .is_err()
                );
            }
        }
    }
    #[tokio::test]
    async fn state_plane_handle_limit_refuses_new_resources_until_drop() {
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 1, 4096),
        );
        let first = host
            .create(compute::Dtype::F32, vec![1, 1])
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            host.create(compute::Dtype::F32, vec![1, 1]).await.unwrap(),
            Err(compute::Error::Quota(_))
        ));
        HostStatePlane::drop(&mut host, first).await.unwrap();
        let next = host
            .create(compute::Dtype::F32, vec![1, 1])
            .await
            .unwrap()
            .unwrap();
        HostStatePlane::drop(&mut host, next).await.unwrap();
        assert_eq!(host.live_handles, 0);
    }
}

#[cfg(all(test, target_os = "macos"))]
#[path = "profile_tests.rs"]
mod profile_tests;
