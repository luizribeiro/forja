use super::{
    Arc, BackendError, Cell, CommandList, CommandResources, CommitResidency, Completion,
    CompletionAccess, DispatchProfile, Duration, GpuTimestamps, GraphArenaReservation, Instant,
    InternalDispatchProfile, MTL4CommandBuffer, MetalBackend, MetalSubmission, Mutex,
    PreparedMetalGraph, ProfileTimestampGranularity, ProtocolObject, RefCell, Retained, Submission,
    SubmissionObjects, SubmissionProfile,
};
#[cfg(test)]
use super::{DType, LAST_SUBMIT_NANOS, Op, store_duration};

thread_local! { static OWNERS: RefCell<Vec<forja_core::CompletionResource>> = const { RefCell::new(Vec::new()) }; }
thread_local! { static CALL_TIMEOUT: Cell<Option<Duration>> = const { Cell::new(None) }; }
pub(crate) struct TimeoutScope(Option<Duration>);
impl TimeoutScope {
    pub(crate) fn enter(timeout: Duration) -> Self {
        Self(CALL_TIMEOUT.replace(Some(timeout)))
    }
}
impl Drop for TimeoutScope {
    fn drop(&mut self) {
        CALL_TIMEOUT.set(self.0);
    }
}

pub(crate) struct SubmissionOwners(Vec<forja_core::CompletionResource>);
impl SubmissionOwners {
    pub(crate) fn enter(owners: Vec<forja_core::CompletionResource>) -> Self {
        Self(OWNERS.replace(owners))
    }
}
impl Drop for SubmissionOwners {
    fn drop(&mut self) {
        OWNERS.set(std::mem::take(&mut self.0));
    }
}
pub(super) fn retain_owners(completion: &Completion) {
    OWNERS.with(|owners| completion.retain_resources(owners.borrow().clone()));
}

pub(super) struct PreparedBuffer {
    pub(super) observation: Option<crate::execution_measurement::Observation>,
    pub(super) predictions: Vec<Duration>,
    pub(super) reserve: Duration,
    pub(super) command_buffer: Retained<ProtocolObject<dyn MTL4CommandBuffer>>,
    pub(super) accesses: Vec<CompletionAccess>,
    pub(super) resources: CommandResources,
    pub(super) residency: CommitResidency,
    pub(super) timestamps: GpuTimestamps,
    pub(super) objects: SubmissionObjects,
    pub(super) profile: Option<SubmissionProfile>,
    pub(super) dispatch_profiles: Vec<DispatchProfile>,
    pub(super) internal_profiles: Vec<InternalDispatchProfile>,
}
impl PreparedBuffer {
    fn submit(
        mut self,
        backend: &MetalBackend,
        deadline: Instant,
    ) -> Result<MetalSubmission, BackendError> {
        let started = Instant::now();
        let mut submission = backend.commit(
            &self.command_buffer,
            &self.accesses,
            self.resources,
            self.residency,
            self.timestamps,
            &mut self.objects,
        )?;
        submission.observation = self.observation;
        submission.deadline = deadline;
        if let Some(mut profile) = self.profile {
            profile.commit = started.elapsed();
            submission.profile = Some(Mutex::new(profile));
            submission.dispatch_profiles = self.dispatch_profiles;
            submission.internal_profiles = self.internal_profiles;
        }
        Ok(submission)
    }
}

fn partition(
    predictions: &[Duration],
    limits: forja_core::GpuWorkLimits,
    reserve: Duration,
) -> Result<Vec<std::ops::Range<usize>>, BackendError> {
    let mut ranges = Vec::new();
    if reserve >= limits.command_buffer() {
        return Err(BackendError::QuotaExceeded);
    }
    let (mut start, mut buffer, mut total) = (0, reserve, reserve);
    for (index, &time) in predictions.iter().enumerate() {
        if time > limits.dispatch() {
            return Err(BackendError::QuotaExceeded);
        }
        total = total.checked_add(time).ok_or(BackendError::QuotaExceeded)?;
        if total > limits.submission() {
            return Err(BackendError::QuotaExceeded);
        }
        if buffer
            .checked_add(time)
            .is_none_or(|sum| sum > limits.command_buffer())
        {
            ranges.push(start..index);
            start = index;
            buffer = reserve;
            total = total
                .checked_add(reserve)
                .ok_or(BackendError::QuotaExceeded)?;
        }
        buffer = buffer
            .checked_add(time)
            .filter(|sum| *sum <= limits.command_buffer())
            .ok_or(BackendError::QuotaExceeded)?;
        if total > limits.submission() {
            return Err(BackendError::QuotaExceeded);
        }
    }
    ranges.push(start..predictions.len());
    Ok(ranges)
}

impl MetalBackend {
    pub(crate) fn command_timeout(&self) -> Duration {
        if crate::work_admission::calibrating() {
            self.gpu_timeout
        } else {
            CALL_TIMEOUT
                .get()
                .map_or(self.gpu_timeout, |timeout| timeout.min(self.gpu_timeout))
        }
    }

    fn submission_limits(&self) -> Result<forja_core::GpuWorkLimits, BackendError> {
        let base = self.work_limits;
        let Some(timeout) = CALL_TIMEOUT
            .get()
            .filter(|&timeout| !crate::work_admission::calibrating() && timeout < base.timeout())
        else {
            return Ok(base);
        };
        let limits = forja_core::GpuWorkLimits::for_timeout(timeout)
            .map_err(|_| BackendError::QuotaExceeded)?;
        let command = limits
            .command_buffer()
            .min(base.command_buffer())
            .min(timeout / base.backstop_ratio() / 2);
        forja_core::GpuWorkLimits::new(
            limits.dispatch().min(base.dispatch()).min(command),
            command,
            limits.submission().min(base.submission()),
            timeout,
            base.backstop_ratio(),
        )
        .map_err(|_| BackendError::QuotaExceeded)
    }

    fn split_oversized_outputs(
        &self,
        commands: &CommandList,
        predictions: &[Duration],
        reserve: Duration,
        limits: forja_core::GpuWorkLimits,
    ) -> Result<Option<CommandList>, BackendError> {
        let budget = limits
            .dispatch()
            .min(limits.command_buffer().saturating_sub(reserve));
        if !predictions.iter().any(|&p| p > budget) {
            return Ok(None);
        }
        let total = predictions.iter().try_fold(Duration::ZERO, |sum, &p| {
            sum.checked_add(p).ok_or(BackendError::QuotaExceeded)
        })?;
        if total > limits.submission() {
            return Err(BackendError::QuotaExceeded);
        }
        let mut split = CommandList::new();
        for (dispatch, &time) in commands.dispatches().iter().zip(predictions) {
            if time <= budget {
                split.record_validated(dispatch.clone());
                continue;
            }
            let halves = dispatch
                .split_output_rows()
                .map_err(|_| BackendError::InvalidInput)?
                .ok_or(BackendError::QuotaExceeded)?;
            for half in halves {
                if let Some(&variant) = half.backend_data::<crate::ValidatedVariant>() {
                    self.validate_variant(&half, variant.name())
                        .map_err(|_| BackendError::QuotaExceeded)?;
                }
                split.record_validated(half);
            }
            if split.len() > 4096 {
                return Err(BackendError::QuotaExceeded);
            }
        }
        if split.len() > 4096 {
            return Err(BackendError::QuotaExceeded);
        }
        Ok(Some(split))
    }

    #[allow(clippy::too_many_lines)]
    pub(super) fn submit_commands_inner<const PROFILE: bool>(
        &self,
        commands: CommandList,
        graph: Option<&PreparedMetalGraph>,
        graph_arena: Option<&GraphArenaReservation>,
        granularity: ProfileTimestampGranularity,
        cache_nodes: Option<&[Arc<super::work_prediction::NodeCache>]>,
        execution_plan: Option<&Arc<crate::execution_plan::ExecutionPlan>>,
    ) -> Result<MetalSubmission, BackendError> {
        let cache_nodes = cache_nodes.or_else(|| graph.map(|graph| graph.predictions.as_slice()));
        let execution_plan =
            execution_plan.or_else(|| graph.and_then(|graph| graph.execution_plan.as_ref()));
        let limits = self.submission_limits()?;
        let started = Instant::now();
        let first = self.prepare_commands::<PROFILE>(
            commands.clone(),
            graph,
            graph_arena,
            granularity,
            cache_nodes,
            execution_plan.map(|plan| plan.total),
        )?;
        if !self.enforce_work_limits {
            let deadline = Instant::now()
                .checked_add(self.command_timeout())
                .ok_or(BackendError::InvalidInput)?;
            let result = first.submit(self, deadline);
            #[cfg(test)]
            store_duration(&LAST_SUBMIT_NANOS, started.elapsed());
            return result;
        }
        if let Some(split) =
            self.split_oversized_outputs(&commands, &first.predictions, first.reserve, limits)?
        {
            let budget = limits
                .dispatch()
                .min(limits.command_buffer().saturating_sub(first.reserve));
            let graph = graph
                .map(|graph| graph.split_outputs(&first.predictions, budget, &split))
                .transpose()?;
            let split_cache = if graph.is_none() {
                cache_nodes
                    .map(|nodes| {
                        super::work_prediction::split_nodes(nodes, &first.predictions, budget)
                    })
                    .transpose()?
            } else {
                None
            };
            let split_execution =
                execution_plan.map(|plan| plan.split(&split, &first.predictions, budget));
            drop(first);
            return self.submit_commands_inner::<PROFILE>(
                split,
                graph.as_ref(),
                graph_arena,
                granularity,
                split_cache.as_deref(),
                split_execution.as_ref(),
            );
        }
        let ranges = partition(&first.predictions, limits, first.reserve)?;
        let deadline = || {
            Instant::now()
                .checked_add(limits.timeout())
                .ok_or(BackendError::InvalidInput)
        };
        let result = if ranges.len() == 1 {
            first.submit(self, deadline()?)
        } else {
            let predictions = ranges
                .iter()
                .map(|range| {
                    first
                        .predictions
                        .get(range.clone())
                        .ok_or(BackendError::InvalidInput)?
                        .iter()
                        .try_fold(first.reserve, |sum, &p| {
                            sum.checked_add(p).ok_or(BackendError::QuotaExceeded)
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            drop(first);
            let recording = commands.program_recording();
            let dispatches = commands.into_dispatches();
            let chunks = ranges
                .into_iter()
                .map(|range| {
                    let caches = cache_nodes
                        .map(|nodes| nodes.get(range.clone()).ok_or(BackendError::InvalidInput))
                        .transpose()?;
                    let prepared = graph.map(|graph| graph.window(range.clone())).transpose()?;
                    let execution = execution_plan.map(|plan| plan.window(range.clone()));
                    let mut chunk = CommandList::new();
                    for dispatch in dispatches.get(range).ok_or(BackendError::InvalidInput)? {
                        chunk.record_validated(dispatch.clone());
                    }
                    Ok((chunk, prepared, caches, execution))
                })
                .collect::<Result<Vec<_>, BackendError>>()?;
            let overhead = started.elapsed();
            let deadline = deadline()?;
            let mut prefixes = Vec::new();
            let last = chunks.len() - 1;
            for (index, ((chunk, graph, caches, execution), forecast)) in
                chunks.into_iter().zip(predictions).enumerate()
            {
                let committed = (|| {
                    let mut prepared = self.prepare_commands::<PROFILE>(
                        chunk,
                        graph.as_ref(),
                        graph_arena,
                        granularity,
                        caches,
                        execution.as_ref().map(|plan| plan.total),
                    )?;
                    let actual = prepared
                        .predictions
                        .iter()
                        .try_fold(prepared.reserve, |sum, &p| {
                            sum.checked_add(p).ok_or(BackendError::QuotaExceeded)
                        })?;
                    if actual > forecast || Instant::now() >= deadline {
                        return Err(BackendError::QuotaExceeded);
                    }
                    if index == 0
                        && let Some(profile) = &mut prepared.profile
                    {
                        profile.validation += overhead;
                        profile.program_recording = recording;
                    }
                    prepared.submit(self, deadline)
                })();
                let mut submission = match committed {
                    Ok(submission) => submission,
                    Err(error) => return retain_prefix_error(prefixes, error),
                };
                if index == last {
                    submission.prefixes = prefixes;
                    return Ok(submission);
                }
                let result =
                    submission.wait_timeout(deadline.saturating_duration_since(Instant::now()));
                let result = result.and_then(|()| {
                    if let Some(work) = submission.completion.work.get()
                        && work.prediction_failed()
                    {
                        return Err(BackendError::GpuPredictionExceeded {
                            predicted: work.prediction(),
                            measured: submission.completion.gpu_time(),
                            ratio: work.ratio(),
                        });
                    }
                    Ok(())
                });
                prefixes.push(submission);
                if let Err(error) = result {
                    return retain_prefix_error(prefixes, error);
                }
            }
            Err(BackendError::ExecutionFailed)
        };
        #[cfg(test)]
        store_duration(&LAST_SUBMIT_NANOS, started.elapsed());
        result
    }
}

fn retain_prefix_error(
    mut prefixes: Vec<MetalSubmission>,
    error: BackendError,
) -> Result<MetalSubmission, BackendError> {
    let Some(mut submission) = prefixes.pop() else {
        return Err(error);
    };
    submission.failure = Some(error);
    submission.prefixes = prefixes;
    Ok(submission)
}

pub(super) fn merge_profiles(
    profiles: impl Iterator<Item = Option<SubmissionProfile>>,
) -> Option<SubmissionProfile> {
    let mut total = SubmissionProfile::default();
    for profile in profiles {
        let mut profile = profile?;
        for dispatch in &mut profile.per_dispatch {
            dispatch.gpu_start = dispatch.gpu_start.checked_add(total.gpu_time)?;
            dispatch.gpu_end = dispatch.gpu_end.checked_add(total.gpu_time)?;
            for internal in &mut dispatch.internal {
                internal.gpu_start = internal.gpu_start.checked_add(total.gpu_time)?;
                internal.gpu_end = internal.gpu_end.checked_add(total.gpu_time)?;
            }
        }
        macro_rules! add { ($($field:ident),*) => {$(total.$field = total.$field.checked_add(profile.$field)?;)*}; }
        add!(
            predicted_gpu,
            command_buffer_reserve,
            command_buffers,
            program_compile_fallbacks,
            epilogue_library_compilations,
            epilogue_pipeline_specializations,
            epilogue_cache_hits,
            epilogue_compile_fallbacks,
            sample_fallbacks,
            validation,
            residency,
            encoding,
            commit,
            wait,
            gpu_time,
            dispatches,
            barriers
        );
        for (target, source) in [
            (&mut total.program_recording, profile.program_recording),
            (&mut total.program_encoding, profile.program_encoding),
            (&mut total.metadata_buffers, profile.metadata_buffers),
        ] {
            target.count = target.count.checked_add(source.count)?;
            target.time = target.time.checked_add(source.time)?;
        }
        total.work_calibration = profile.work_calibration.or(total.work_calibration);
        total.per_dispatch.extend(profile.per_dispatch);
        total.buffer_admission.extend(profile.buffer_admission);
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn enforcing_backend() -> MetalBackend {
        MetalBackend::new().unwrap().with_work_limits(
            forja_core::GpuWorkLimits::for_timeout(Duration::from_secs(10)).unwrap(),
        )
    }

    #[test]
    fn forecasting_is_observational_until_host_enables_admission() {
        use forja_core::Backend;
        let _rates = crate::work_admission::TestCalibration::fast();
        let backend = MetalBackend::new().unwrap();
        let input = backend.alloc(DType::F32, &[33, 4097]).unwrap();
        let output = backend.alloc(DType::F32, &[33, 4097]).unwrap();
        let bytes = vec![1.0_f32.to_ne_bytes(); 33 * 4097].concat();
        backend.write(&input, &bytes).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();
        for _ in 0..2 {
            let submission = backend.submit_commands_profiled(commands.clone()).unwrap();
            submission.wait().unwrap();
            assert!(submission.profile().unwrap().predicted_gpu > Duration::ZERO);
            assert!(submission.completion.work.get().is_none());
            assert_eq!(backend.read(&output).unwrap(), bytes);
        }
        let backend = backend.with_work_limits(
            forja_core::GpuWorkLimits::new(
                Duration::from_nanos(1),
                Duration::from_nanos(1),
                Duration::from_nanos(1),
                Duration::from_secs(10),
                4,
            )
            .unwrap(),
        );
        let serial = *backend.next_event_value.lock().unwrap();
        assert!(matches!(
            backend.submit_commands(commands),
            Err(BackendError::QuotaExceeded)
        ));
        assert_eq!(*backend.next_event_value.lock().unwrap(), serial);
    }

    #[test]
    fn default_limits_admit_real_attention_chunks_after_query_splitting() {
        let backend = enforcing_backend();
        for (heads, width, dtype, queries, keys, variant) in [
            (
                16_u32,
                128,
                DType::F16,
                4096,
                4096_u32,
                "paged-sdpa.steel-half",
            ),
            (16, 128, DType::F16, 4096, 16384, "paged-sdpa.steel-half"),
            (
                16,
                128,
                DType::F16,
                4096,
                262_144,
                "paged-sdpa.steel-bounded",
            ),
            (
                16,
                128,
                DType::F16,
                1,
                262_144,
                "paged-sdpa.vector-gqa-two-pass-1024",
            ),
            (24, 128, DType::F32, 1, 262_144, "paged-sdpa.steel-bounded"),
            (
                24,
                256,
                DType::F32,
                1,
                262_144,
                "paged-sdpa.steel-bounded-256",
            ),
        ] {
            assert_attention_admitted(&backend, (heads, width, dtype, queries, keys, variant));
        }
    }
    fn assert_attention_admitted(backend: &MetalBackend, case: (u32, u32, DType, u32, u32, &str)) {
        use forja_core::{Backend, BackendDispatchData};
        let (heads, width, dtype, queries, keys, variant) = case;
        let pages = keys.div_ceil(128);
        let input = [
            backend.alloc(DType::F32, &[heads, queries, width]).unwrap(),
            backend.alloc(dtype, &[pages, 128, 8, width]).unwrap(),
            backend.alloc(dtype, &[pages, 128, 8, width]).unwrap(),
            backend.alloc(DType::U32, &[pages, 2]).unwrap(),
            backend.alloc(DType::U32, &[pages]).unwrap(),
        ];
        let output = backend.alloc(DType::F32, &[heads, queries, width]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::PagedSdpa {
                    scale: 1.0,
                    causal: true,
                    q_start: keys - queries,
                    key_length: keys,
                    page_tokens: 128,
                },
                &input.iter().collect::<Vec<_>>(),
                &output,
            )
            .unwrap();
        let selection = backend
            .validate_variant(commands.last_dispatch().unwrap(), variant)
            .unwrap();
        commands.set_last_backend_data(BackendDispatchData::new(selection));
        let buffers = loop {
            let prepared = backend
                .prepare_commands::<false>(
                    commands.clone(),
                    None,
                    None,
                    ProfileTimestampGranularity::Precise,
                    None,
                    None,
                )
                .unwrap();
            if let Some(split) = backend
                .split_oversized_outputs(
                    &commands,
                    &prepared.predictions,
                    prepared.reserve,
                    backend.work_limits,
                )
                .unwrap()
            {
                commands = split;
            } else {
                break partition(&prepared.predictions, backend.work_limits, prepared.reserve)
                    .unwrap()
                    .len();
            }
        };
        assert!(buffers > 0);
        drop(commands);
        for tensor in input.iter().chain(std::iter::once(&output)) {
            backend.release(tensor).unwrap();
        }
    }

    #[test]
    fn partition_checks_dispatch_buffer_and_submission_limits() {
        let limits = forja_core::GpuWorkLimits::new(
            Duration::from_secs(1),
            Duration::from_secs(1),
            Duration::from_secs(8),
            Duration::from_secs(10),
            4,
        )
        .unwrap();
        let ms = Duration::from_millis;
        assert_eq!(
            partition(&[ms(600), ms(401), ms(599)], limits, Duration::ZERO),
            Ok(vec![0..1, 1..3])
        );
        assert_eq!(
            partition(&[ms(1001)], limits, Duration::ZERO),
            Err(BackendError::QuotaExceeded)
        );
        assert_eq!(
            partition(&[ms(1000); 9], limits, Duration::ZERO),
            Err(BackendError::QuotaExceeded)
        );
    }
    #[test]
    fn caller_deadlines_refuse_before_commit_and_do_not_leak_between_calls() {
        use forja_core::Backend;
        let backend = enforcing_backend();
        let input = backend.alloc(DType::F32, &[4097]).unwrap();
        let output = backend.alloc(DType::F32, &[4097]).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();
        backend
            .prepare_dispatch(commands.last_dispatch().unwrap())
            .unwrap();
        let serial = *backend.next_event_value.lock().unwrap();
        assert!(matches!(
            backend.submit_with_timeout(
                commands.clone(),
                Vec::new(),
                false,
                Duration::from_micros(1)
            ),
            Err(BackendError::QuotaExceeded)
        ));
        assert_eq!(*backend.next_event_value.lock().unwrap(), serial);
        backend
            .submit_with_timeout(commands, Vec::new(), false, Duration::from_secs(10))
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(*backend.next_event_value.lock().unwrap(), serial + 1);
    }

    #[test]
    fn caller_deadline_bounds_slot_backpressure_before_commit() {
        use forja_core::Backend;
        let backend = enforcing_backend();
        backend.prepare_work_estimation().unwrap();
        let input = backend.alloc(DType::F32, &[33]).unwrap();
        let output = backend.alloc(DType::F32, &[33]).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();
        backend
            .prepare_dispatch(commands.last_dispatch().unwrap())
            .unwrap();
        let held = (0..super::super::MAX_IN_FLIGHT_SUBMISSIONS)
            .map(|_| backend.in_flight.checkout(&backend, 4096).unwrap())
            .collect::<Vec<_>>();
        let serial = *backend.next_event_value.lock().unwrap();
        assert!(matches!(
            backend.submit_with_timeout(commands, Vec::new(), false, Duration::from_micros(1)),
            Err(BackendError::QuotaExceeded)
        ));
        assert_eq!(*backend.next_event_value.lock().unwrap(), serial);
        drop(held);
    }

    #[test]
    fn buffer_reserve_is_charged_for_every_partition() {
        let limits = forja_core::GpuWorkLimits::new(
            Duration::from_secs(1),
            Duration::from_secs(1),
            Duration::from_secs(8),
            Duration::from_secs(10),
            4,
        )
        .unwrap();
        let ms = Duration::from_millis;
        assert_eq!(
            partition(&[ms(600), ms(401), ms(599)], limits, ms(50)),
            Ok(vec![0..1, 1..2, 2..3])
        );
        assert_eq!(
            partition(&[ms(1000)], limits, ms(50)),
            Err(BackendError::QuotaExceeded)
        );
        assert_eq!(
            partition(&[ms(1); 9], limits, ms(999)),
            Err(BackendError::QuotaExceeded)
        );
    }

    #[test]
    fn failed_prefix_does_not_commit_the_remaining_buffer() {
        use forja_core::Backend;
        let mut backend = enforcing_backend();
        let table = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        let ids = backend.alloc(DType::U32, &[1]).unwrap();
        let output = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        let target = backend.alloc(DType::F32, &[1, 4097]).unwrap();
        backend.write(&ids, &99_u32.to_ne_bytes()).unwrap();
        let sentinel = vec![7_u8; 4097 * 4];
        backend.write(&target, &sentinel).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Embed, &[&table, &ids], &output)
            .unwrap();
        commands.dispatch(Op::Copy, &[&output], &target).unwrap();
        let prepared = backend
            .prepare_commands::<false>(
                commands.clone(),
                None,
                None,
                ProfileTimestampGranularity::Precise,
                None,
                None,
            )
            .unwrap();
        let budget = *prepared.predictions.iter().max().unwrap();
        let reserve = prepared.reserve;
        drop(prepared);
        backend.work_limits = forja_core::GpuWorkLimits::new(
            budget,
            budget + reserve,
            (budget + reserve) * 3,
            Duration::from_secs(10),
            4,
        )
        .unwrap();
        let serial = *backend.next_event_value.lock().unwrap();
        let submission = backend.submit_commands(commands).unwrap();
        assert!(matches!(
            submission.wait(),
            Err(BackendError::DeviceErrorFlag { value: 99, .. })
        ));
        assert!(!submission.gpu_time().unwrap().is_zero());
        assert_eq!(*backend.next_event_value.lock().unwrap(), serial + 1);
        assert_eq!(backend.read(&target).unwrap(), sentinel);
    }

    #[test]
    fn oversized_copy_splits_output_views_without_changing_bits() {
        use forja_core::Backend;
        let mut backend = enforcing_backend();
        let input = backend.alloc(DType::F32, &[33, 4097]).unwrap();
        let output = backend.alloc(DType::F32, &[33, 4097]).unwrap();
        let expected = (0..33 * 4097_u32)
            .flat_map(u32::to_ne_bytes)
            .collect::<Vec<_>>();
        backend.write(&input, &expected).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();
        let prepared = backend
            .prepare_commands::<false>(
                commands.clone(),
                None,
                None,
                ProfileTimestampGranularity::Precise,
                None,
                None,
            )
            .unwrap();
        let original = prepared.predictions[0];
        let reserve = prepared.reserve;
        drop(prepared);
        let budget = original * 3 / 4;
        backend.work_limits = forja_core::GpuWorkLimits::new(
            budget,
            budget + reserve,
            (original + reserve) * 8,
            Duration::from_secs(10),
            4,
        )
        .unwrap();
        let submission = backend.submit_commands_profiled(commands).unwrap();
        submission.wait().unwrap();
        assert_eq!(backend.read(&output).unwrap(), expected);
        let profile = submission.profile().unwrap();
        assert!(profile.command_buffers > 1);
        assert!(
            profile
                .per_dispatch
                .iter()
                .all(|d| d.predicted_gpu <= budget)
        );
    }

    #[test]
    fn incremental_copy_submission_preserves_dependencies_and_profiles() {
        incremental_copy(false);
    }
    #[test]
    fn incremental_replay_keeps_static_plans_and_the_inherited_pipeline() {
        incremental_copy(true);
    }
    fn incremental_copy(replay: bool) {
        use forja_core::Backend;
        let mut backend = enforcing_backend();
        let tensors = (0..4)
            .map(|_| backend.alloc(DType::F32, &[4097]).unwrap())
            .collect::<Vec<_>>();
        let expected = (0..4097_u32)
            .flat_map(|i| f32::from(u16::try_from(i).unwrap()).to_ne_bytes())
            .collect::<Vec<_>>();
        backend.write(&tensors[0], &expected).unwrap();
        let mut commands = CommandList::new();
        for pair in tensors.windows(2) {
            commands.dispatch(Op::Copy, &[&pair[0]], &pair[1]).unwrap();
        }
        let graph = replay.then(|| {
            let mut template = forja_core::GraphTemplate::new(
                forja_core::ParamSpace::new(Vec::new()).unwrap(),
                forja_core::GraphLimits::default(),
            );
            for pair in tensors.windows(2) {
                template
                    .dispatch(
                        Op::Copy,
                        &[&pair[0].clone().into()],
                        &pair[1].clone().into(),
                    )
                    .unwrap();
            }
            backend.prepare_graph(template).unwrap()
        });
        let state = graph
            .as_ref()
            .and_then(forja_core::PreparedGraph::backend_state::<PreparedMetalGraph>);
        assert_split_plans(state);
        let prepared = backend
            .prepare_commands::<false>(
                commands.clone(),
                state,
                None,
                ProfileTimestampGranularity::Precise,
                None,
                None,
            )
            .unwrap();
        let unit = prepared.predictions[0];
        let reserve = prepared.reserve;
        drop(prepared);
        let dispatch_budget = unit + unit / 2;
        let buffer = dispatch_budget + reserve;
        backend.work_limits = forja_core::GpuWorkLimits::new(
            dispatch_budget,
            buffer,
            (unit + reserve) * 4,
            Duration::from_secs(10),
            4,
        )
        .unwrap();
        let before = *backend.next_event_value.lock().unwrap();
        let predictions = forecast_count(state);
        let lookups = backend
            .pipelines
            .lookups
            .load(super::super::Ordering::Relaxed);
        let submission = graph
            .as_ref()
            .map_or_else(
                || backend.submit_commands_profiled(commands),
                |graph| backend.replay_graph(graph, Vec::new(), true, None),
            )
            .unwrap();
        submission.wait().unwrap();
        assert_eq!(*backend.next_event_value.lock().unwrap(), before + 3);
        if replay {
            assert_eq!(predictions, Some(3));
            assert_eq!(forecast_count(state), predictions);
            assert_eq!(
                backend
                    .pipelines
                    .lookups
                    .load(super::super::Ordering::Relaxed),
                lookups
            );
        }
        assert_eq!(backend.read(&tensors[3]).unwrap(), expected);
        let profile = submission.profile().unwrap();
        assert_incremental_profile(&profile, unit, reserve);
        assert_eq!(Some(profile.gpu_time), submission.gpu_time());
        if let Some(graph) = &graph {
            assert_rebound_prediction_hits(&backend, graph, &tensors[0], &tensors[3]);
        }
    }
    fn assert_split_plans(state: Option<&PreparedMetalGraph>) {
        if let Some(state) = state {
            let expanded = state
                .split_outputs(
                    &[Duration::from_nanos(1), Duration::ZERO, Duration::ZERO],
                    Duration::ZERO,
                    &CommandList::new(),
                )
                .unwrap();
            let plans = &expanded.encoding.as_ref().unwrap().dispatches;
            assert!(plans[0].is_none() && plans[1].is_none());
            assert!(matches!(
                plans[2].as_ref().unwrap().calls.first(),
                Some(super::super::PlanCall::Pipeline(_))
            ));
        }
    }
    fn assert_rebound_prediction_hits(
        backend: &MetalBackend,
        graph: &forja_core::PreparedGraph,
        input: &forja_core::Tensor,
        output: &forja_core::Tensor,
    ) {
        use forja_core::Backend;
        let state = graph.backend_state::<PreparedMetalGraph>().unwrap();
        let before = forecast_count(Some(state));
        let replacement = backend.alloc(DType::F32, &[4097]).unwrap();
        let changed = vec![17_u8; 4097 * 4];
        backend.write(&replacement, &changed).unwrap();
        let mut bindings = forja_core::ReplayBindings::default();
        bindings.view(input.clone(), replacement).unwrap();
        let submission = backend
            .replay_graph(graph, Vec::new(), true, Some(&bindings))
            .unwrap();
        submission.wait().unwrap();
        assert_eq!(forecast_count(Some(state)), before);
        assert_eq!(backend.read(output).unwrap(), changed);
    }
    fn forecast_count(state: Option<&PreparedMetalGraph>) -> Option<usize> {
        state.map(|state| {
            state
                .predictions
                .iter()
                .map(|cache| cache.calculations.load(super::super::Ordering::Relaxed))
                .sum()
        })
    }
    fn assert_incremental_profile(profile: &SubmissionProfile, unit: Duration, reserve: Duration) {
        assert_eq!(profile.command_buffers, 3);
        assert_eq!(profile.per_dispatch.len(), 3);
        assert_eq!(profile.predicted_gpu, (unit + reserve) * 3);
        assert_eq!(profile.buffer_admission.len(), 3);
        assert!(
            profile
                .buffer_admission
                .iter()
                .all(|buffer| buffer.predicted_gpu == unit + reserve
                    && buffer.reserve == reserve
                    && buffer.backstop_ratio == 4
                    && !buffer.gpu_time.is_zero())
        );
        assert_eq!(
            profile
                .buffer_admission
                .iter()
                .map(|buffer| buffer.gpu_time)
                .sum::<Duration>(),
            profile.gpu_time
        );
    }
}
