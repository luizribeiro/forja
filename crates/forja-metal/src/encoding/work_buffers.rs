use super::{
    BackendError, CommandList, CommandResources, CommitResidency, Completion, CompletionAccess,
    DispatchProfile, Duration, GpuTimestamps, GraphArenaReservation, Instant,
    InternalDispatchProfile, MTL4CommandBuffer, MetalBackend, MetalSubmission, Mutex,
    PreparedMetalGraph, ProfileTimestampGranularity, ProtocolObject, RefCell, Retained, Submission,
    SubmissionObjects, SubmissionProfile,
};
#[cfg(test)]
use super::{DType, LAST_SUBMIT_NANOS, Op, store_duration};

thread_local! { static OWNERS: RefCell<Vec<forja_core::CompletionResource>> = const { RefCell::new(Vec::new()) }; }
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
    fn split_oversized_outputs(
        &self,
        commands: &CommandList,
        predictions: &[Duration],
        reserve: Duration,
    ) -> Result<Option<CommandList>, BackendError> {
        let budget = self
            .work_limits
            .dispatch()
            .min(self.work_limits.command_buffer().saturating_sub(reserve));
        if !predictions.iter().any(|&p| p > budget) {
            return Ok(None);
        }
        let total = predictions.iter().try_fold(Duration::ZERO, |sum, &p| {
            sum.checked_add(p).ok_or(BackendError::QuotaExceeded)
        })?;
        if total > self.work_limits.submission() {
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
    ) -> Result<MetalSubmission, BackendError> {
        let started = Instant::now();
        let first =
            self.prepare_commands::<PROFILE>(commands.clone(), graph, graph_arena, granularity)?;
        if let Some(split) =
            self.split_oversized_outputs(&commands, &first.predictions, first.reserve)?
        {
            drop(first);
            return self.submit_commands_inner::<PROFILE>(split, None, graph_arena, granularity);
        }
        let ranges = partition(&first.predictions, self.work_limits, first.reserve)?;
        let deadline = || {
            Instant::now()
                .checked_add(self.gpu_timeout)
                .ok_or(BackendError::InvalidInput)
        };
        let result = if ranges.len() == 1 {
            first.submit(self, deadline()?)
        } else {
            drop(first);
            let recording = commands.program_recording();
            let dispatches = commands.into_dispatches();
            let chunks = ranges
                .into_iter()
                .map(|range| {
                    let mut chunk = CommandList::new();
                    for dispatch in &dispatches[range] {
                        chunk.record_validated(dispatch.clone());
                    }
                    chunk
                })
                .collect::<Vec<_>>();
            let mut predictions = Vec::new();
            for chunk in &chunks {
                let prepared = self.prepare_commands::<PROFILE>(
                    chunk.clone(),
                    None,
                    graph_arena,
                    granularity,
                )?;
                if partition(&prepared.predictions, self.work_limits, prepared.reserve)?.len() != 1
                {
                    return Err(BackendError::QuotaExceeded);
                }
                predictions.push(
                    prepared
                        .predictions
                        .iter()
                        .try_fold(prepared.reserve, |sum, &p| {
                            sum.checked_add(p).ok_or(BackendError::QuotaExceeded)
                        })?,
                );
            }
            let total = predictions.iter().try_fold(Duration::ZERO, |sum, &p| {
                sum.checked_add(p).ok_or(BackendError::QuotaExceeded)
            })?;
            if total > self.work_limits.submission() {
                return Err(BackendError::QuotaExceeded);
            }
            let overhead = started.elapsed();
            let deadline = deadline()?;
            let mut prefixes = Vec::new();
            let last = chunks.len() - 1;
            for (index, (chunk, forecast)) in chunks.into_iter().zip(predictions).enumerate() {
                let mut prepared =
                    self.prepare_commands::<PROFILE>(chunk, None, graph_arena, granularity)?;
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
                let mut submission = prepared.submit(self, deadline)?;
                if index == last {
                    submission.prefixes = prefixes;
                    return Ok(submission);
                }
                submission.wait_timeout(deadline.saturating_duration_since(Instant::now()))?;
                if let Some(work) = submission.completion.work.get()
                    && work.prediction_failed()
                {
                    return Err(BackendError::GpuPredictionExceeded {
                        predicted: work.prediction(),
                        measured: submission.completion.gpu_time(),
                        ratio: work.ratio(),
                    });
                }
                prefixes.push(submission);
            }
            Err(BackendError::ExecutionFailed)
        };
        #[cfg(test)]
        store_duration(&LAST_SUBMIT_NANOS, started.elapsed());
        result
    }
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
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;
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
        let mut backend = MetalBackend::new().unwrap();
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
        assert!(matches!(
            backend.submit_commands(commands),
            Err(BackendError::DeviceErrorFlag { value: 99, .. })
        ));
        assert_eq!(*backend.next_event_value.lock().unwrap(), serial + 1);
        assert_eq!(backend.read(&target).unwrap(), sentinel);
    }

    #[test]
    fn oversized_copy_splits_output_views_without_changing_bits() {
        use forja_core::Backend;
        let mut backend = MetalBackend::new().unwrap();
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
        use forja_core::Backend;
        let mut backend = MetalBackend::new().unwrap();
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
        let prepared = backend
            .prepare_commands::<false>(
                commands.clone(),
                None,
                None,
                ProfileTimestampGranularity::Precise,
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
        let submission = backend.submit_commands_profiled(commands).unwrap();
        submission.wait().unwrap();
        assert_eq!(*backend.next_event_value.lock().unwrap(), before + 3);
        assert_eq!(backend.read(&tensors[3]).unwrap(), expected);
        let profile = submission.profile().unwrap();
        assert_eq!(profile.command_buffers, 3);
        assert_eq!(profile.per_dispatch.len(), 3);
        assert_eq!(profile.predicted_gpu, (unit + reserve) * 3);
        assert_eq!(Some(profile.gpu_time), submission.gpu_time());
    }
}
