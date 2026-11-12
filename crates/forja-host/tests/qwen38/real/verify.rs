use super::{Error, Runner, component, load_configured};
use forja_draft_testing::{
    ExactnessChecks, TestDrafter, Trace, assert_draft_exact, sampling_modes,
};
use forja_host::{
    BatchPhase, BatchSelection, BatchShape, EngineBatch, EngineBatchLane, EngineLaneCheckpoint,
    HybridPrefixStore, SamplingParams, SnapshotAllowance,
    speculative::{VerifyAcceptance, VerifyBudget, VerifyCut, select_verify_shape},
};
use forja_serve::{DraftSource, PromptLookup};
use std::{collections::HashMap, sync::atomic::AtomicBool};

struct Request {
    runner: Runner,
    cache: HybridPrefixStore<forja_metal::MetalBackend>,
    prefix: Vec<u32>,
}
impl Request {
    async fn new(position: u32) -> Result<Self, Box<dyn Error>> {
        let (runner, _) = load_configured(true, None, Some(8192)).await?;
        let namespace = runner.canonical_prefix_namespace([7; 32], &["verify-exact"])?;
        let cache = runner
            .create_canonical_prefix_store(namespace, SnapshotAllowance::new(2 << 30, None)?)?;
        let mut request = Self {
            runner,
            cache,
            prefix: (0..position).map(|row| 100 + row % 8).collect(),
        };
        request.reset_to(0).await?;
        let root = (position - 1) / 64 * 64;
        request.prefill(0, root).await?;
        if root > 0 {
            let entry = request
                .runner
                .capture_canonical_prefix(0, SnapshotAllowance::new(2 << 30, None)?)?;
            request.cache.publish(entry, &AtomicBool::new(false))?;
        }
        Ok(request)
    }
    async fn reset_to(&mut self, end: u32) -> Result<(), Box<dyn Error>> {
        self.runner.prepare_paged_lane(0, None, 0)?;
        let namespace = self.cache.namespace();
        let hit = self.cache.lookup(namespace, &self.prefix)?;
        let prompt = self
            .runner
            .canonical_prompt(namespace, self.prefix.clone())?;
        let start = self
            .runner
            .prepare_canonical_lane(0, prompt, hit.as_ref(), 8192)?;
        self.prefill(u32::try_from(start)?, end).await
    }
    async fn prefill(&mut self, mut start: u32, end: u32) -> Result<(), Box<dyn Error>> {
        while start < end {
            let count = self
                .runner
                .prompt_piece_limit(start, (end - start).min(2048))?;
            let shape = BatchShape {
                phase: BatchPhase::Prompt,
                bucket: 1,
                tokens: [64, 128, 512, 2048]
                    .into_iter()
                    .find(|size| *size >= count)
                    .unwrap(),
                selection: BatchSelection::Combined,
            };
            let output = self
                .runner
                .batch_step(batch(
                    shape,
                    self.prefix[usize::try_from(start)?..usize::try_from(start + count)?].to_vec(),
                    start,
                    SamplingParams::default(),
                ))
                .await??;
            self.runner.read_queued_batch_tokens(output).await?;
            start += count;
        }
        Ok(())
    }
    fn assert_position(&self, position: u64) -> Result<(), Box<dyn Error>> {
        assert_eq!(
            self.runner
                .checkpoint_state_lane(0)?
                .unwrap()
                .position()
                .tokens,
            position
        );
        assert_eq!(
            self.runner.paged_lane_prefix(0)?.unwrap().logical_len()?,
            position
        );
        Ok(())
    }
    async fn decode(
        &mut self,
        pending: Option<u32>,
        position: u32,
        sampling: SamplingParams,
    ) -> Result<u32, Box<dyn Error>> {
        let output = self
            .runner
            .enqueue_batch_step(EngineBatch {
                shape: BatchShape {
                    phase: BatchPhase::Decode,
                    bucket: 1,
                    tokens: 1,
                    selection: BatchSelection::Combined,
                },
                lanes: vec![Some(EngineBatchLane {
                    slot: 0,
                    tokens: pending.map(|token| vec![token]),
                    start_pos: position,
                    sampling,
                })],
            })
            .await??;
        let bytes = self.runner.read_queued_batch_tokens(output).await?;
        Ok(u32::from_le_bytes(
            bytes.try_into().map_err(|_| "invalid sample")?,
        ))
    }
}
fn batch(
    shape: BatchShape,
    tokens: Vec<u32>,
    start_pos: u32,
    sampling: SamplingParams,
) -> EngineBatch {
    EngineBatch {
        shape,
        lanes: vec![Some(EngineBatchLane {
            slot: 0,
            tokens: Some(tokens),
            start_pos,
            sampling,
        })],
    }
}

struct Exactness {
    request: Request,
    targets: HashMap<Vec<u32>, u32>,
    cap: u32,
    lookup: Option<forja_config::LookupPolicy>,
}
impl Exactness {
    async fn propose(
        &mut self,
        history: &[u32],
        count: usize,
        sampling: SamplingParams,
        drafter: &mut TestDrafter,
    ) -> Result<Vec<u32>, Box<dyn Error>> {
        loop {
            let mut trial = *drafter;
            let mut missing = None;
            let proposals = trial.propose(history, count, 248_320, |prefix| {
                self.targets.get(prefix).copied().unwrap_or_else(|| {
                    missing.get_or_insert_with(|| prefix.to_vec());
                    0
                })
            });
            let Some(prefix) = missing else {
                *drafter = trial;
                return Ok(proposals);
            };
            assert!(prefix.starts_with(history));
            let start = history.len() - 1;
            let state = self.request.runner.checkpoint_state_lane(0)?.unwrap();
            let mut checkpoint = EngineLaneCheckpoint::new(
                u32::try_from(state.position().sampler)?,
                *history.last().unwrap(),
                (),
                8192,
                self.request.runner.checkpoint_paged_lane(0)?.unwrap(),
            );
            checkpoint.state_version = Some(state);
            for (row, &input) in prefix[start..].iter().enumerate() {
                let token = self
                    .request
                    .decode(Some(input), u32::try_from(start + row)?, sampling)
                    .await?;
                self.targets
                    .entry(prefix[..=start + row].to_vec())
                    .or_insert(token);
            }
            self.request.runner.rollback_lane(0, checkpoint)?;
        }
    }

    async fn run(
        &mut self,
        sampling: SamplingParams,
        mut drafter: Option<TestDrafter>,
    ) -> Result<Trace, Box<dyn Error>> {
        if drafter.is_none() {
            self.targets.clear();
        }
        let start = u32::try_from(self.request.prefix.len())?;
        self.request.reset_to(start).await?;
        let mut history = self.request.prefix.clone();
        history.push(100 + start % 8);
        let mut trace = Trace::default();
        let limit = 10;
        let shapes = verify_shapes(self.cap);
        let mut lookup = drafter.and(self.lookup.clone()).map(PromptLookup::new);
        let mut explicit = true;
        assert!(self.request.runner.try_reserve_draft_lane(0)?);
        while trace.tokens.len() < limit {
            let position = u32::try_from(history.len() - 1)?;
            let left = limit - trace.tokens.len();
            let count = usize::try_from(self.cap)?.min(left.saturating_sub(1));
            let (proposals, rule_cap) = if let Some(source) = &mut lookup {
                let proposal = source.propose(&history, position, sampling);
                (proposal.tokens, proposal.rule_cap.min(self.cap))
            } else if let Some(drafter) = &mut drafter {
                (
                    self.propose(&history, count, sampling, drafter).await?,
                    self.cap,
                )
            } else {
                (Vec::new(), 0)
            };
            let budget = VerifyBudget::new(
                u64::from(position),
                rule_cap,
                u64::try_from(left)?,
                8192,
                8192,
            )
            .unwrap();
            if let Some(fed) = select_verify_shape(&shapes, &proposals, budget) {
                let mut tokens = vec![*history.last().unwrap()];
                tokens.extend_from_slice(fed.tokens());
                let output = self
                    .request
                    .runner
                    .enqueue_batch_step(batch(fed.shape(), tokens, position, sampling))
                    .await??;
                let bytes = self.request.runner.read(&output.tokens).await?;
                let samples = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(u32::from_le_bytes)
                    .collect::<Vec<_>>();
                trace.shapes.push(fed.shape().tokens);
                let decision = VerifyAcceptance::new(fed, &samples, 248_320)?
                    .scan(|_| Ok::<_, std::convert::Infallible>(None))?;
                trace.accepted.push(decision.accepted());
                if let Some(source) = &mut lookup {
                    source.observe_acceptance(decision.accepted());
                }
                trace.tokens.extend_from_slice(decision.tokens());
                history.extend_from_slice(decision.tokens());
                self.request
                    .runner
                    .publish_verified_batch(output, &decision)?;
                explicit = true;
            } else {
                let token = self
                    .request
                    .decode(
                        explicit.then(|| *history.last().unwrap()),
                        position,
                        sampling,
                    )
                    .await?;
                self.targets.insert(history.clone(), token);
                history.push(token);
                trace.tokens.push(token);
                trace.plain += 1;
                explicit = false;
            }
            self.request
                .assert_position(u64::try_from(history.len() - 1)?)?;
        }
        if self.lookup.is_some() && drafter.is_some() {
            assert_eq!(trace.shapes.first().copied(), Some(self.cap + 1));
        }
        trace.cut = Some(VerifyCut::MaxTokens);
        self.request.runner.try_release_draft_lane(0)?;
        self.request.runner.prepare_paged_lane(0, None, 0)?;
        Ok(trace)
    }
}
fn verify_shapes(cap: u32) -> Vec<BatchShape> {
    (2..=cap + 1)
        .map(|tokens| BatchShape {
            phase: BatchPhase::Verify,
            bucket: 1,
            tokens,
            selection: BatchSelection::Combined,
        })
        .collect()
}
fn exact(position: u32, cap: u32, lookup: bool, group: usize) -> Result<(), Box<dyn Error>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let request = runtime.block_on(Request::new(position))?;
    let profile = forja_config::read_profile(&std::fs::read(component())?)?;
    let policy = if lookup {
        Some(
            profile
                .prompt_lookup()
                .ok_or("lookup policy missing")?
                .clone(),
        )
    } else {
        None
    };
    let mut exactness = Exactness {
        request,
        targets: HashMap::new(),
        cap,
        lookup: policy,
    };
    let checks = if lookup {
        ExactnessChecks::source(true)
    } else {
        true.into()
    };
    assert_draft_exact(
        checks.for_sampling(sampling_modes()[group]),
        |sampling, drafter| runtime.block_on(exactness.run(sampling, drafter)).unwrap(),
    );
    Ok(())
}
macro_rules! exact_tests {
    ($($name:ident: ($position:expr, $cap:expr, $lookup:expr, $group:expr)),+ $(,)?) => {$ (
        #[test]
        #[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
        fn $name() -> Result<(), Box<dyn Error>> { exact($position, $cap, $lookup, $group) }
    )+};
}
exact_tests! {
    metal_real_verify_one_draft_matches_short_position_streams_group_0: (254, 1, false, 0),
    metal_real_verify_one_draft_matches_short_position_streams_group_1: (254, 1, false, 1),
    metal_real_verify_one_draft_matches_short_position_streams_group_2: (254, 1, false, 2),
    metal_real_verify_one_draft_matches_short_position_streams_group_3: (254, 1, false, 3),
    metal_real_verify_one_draft_matches_long_position_streams_group_0: (4097, 1, false, 0),
    metal_real_verify_one_draft_matches_long_position_streams_group_1: (4097, 1, false, 1),
    metal_real_verify_one_draft_matches_long_position_streams_group_2: (4097, 1, false, 2),
    metal_real_verify_one_draft_matches_long_position_streams_group_3: (4097, 1, false, 3),
    metal_real_verify_two_drafts_match_short_position_streams_group_0: (254, 2, false, 0),
    metal_real_verify_two_drafts_match_short_position_streams_group_1: (254, 2, false, 1),
    metal_real_verify_two_drafts_match_short_position_streams_group_2: (254, 2, false, 2),
    metal_real_verify_two_drafts_match_short_position_streams_group_3: (254, 2, false, 3),
    metal_real_verify_two_drafts_match_long_position_streams_group_0: (4097, 2, false, 0),
    metal_real_verify_two_drafts_match_long_position_streams_group_1: (4097, 2, false, 1),
    metal_real_verify_two_drafts_match_long_position_streams_group_2: (4097, 2, false, 2),
    metal_real_verify_two_drafts_match_long_position_streams_group_3: (4097, 2, false, 3),
    metal_real_lookup_one_draft_matches_short_position_streams_group_0: (254, 1, true, 0),
    metal_real_lookup_one_draft_matches_short_position_streams_group_1: (254, 1, true, 1),
    metal_real_lookup_one_draft_matches_short_position_streams_group_2: (254, 1, true, 2),
    metal_real_lookup_one_draft_matches_short_position_streams_group_3: (254, 1, true, 3),
    metal_real_lookup_one_draft_matches_long_position_streams_group_0: (4097, 1, true, 0),
    metal_real_lookup_one_draft_matches_long_position_streams_group_1: (4097, 1, true, 1),
    metal_real_lookup_one_draft_matches_long_position_streams_group_2: (4097, 1, true, 2),
    metal_real_lookup_one_draft_matches_long_position_streams_group_3: (4097, 1, true, 3),
    metal_real_lookup_two_drafts_match_short_position_streams_group_0: (254, 2, true, 0),
    metal_real_lookup_two_drafts_match_short_position_streams_group_1: (254, 2, true, 1),
    metal_real_lookup_two_drafts_match_short_position_streams_group_2: (254, 2, true, 2),
    metal_real_lookup_two_drafts_match_short_position_streams_group_3: (254, 2, true, 3),
    metal_real_lookup_two_drafts_match_long_position_streams_group_0: (4097, 2, true, 0),
    metal_real_lookup_two_drafts_match_long_position_streams_group_1: (4097, 2, true, 1),
    metal_real_lookup_two_drafts_match_long_position_streams_group_2: (4097, 2, true, 2),
    metal_real_lookup_two_drafts_match_long_position_streams_group_3: (4097, 2, true, 3),
}
