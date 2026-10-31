//! Minimal engine component exported through the SDK macro.

use forja_sdk::{
    BatchInput, BatchPhase, BatchSelection, BatchShape, BeginDecodeInput, CachePlane, DecodeInput,
    DecodeOutput, Dim, Engine, EngineInfo, EngineLoadConfig, Graph, KvCacheFormat, PagedCacheSet,
    PagedPool, Param, Result, ServingInfo, StepInput, StepOutput, Tensor, Weights, export_engine,
    nn::blocks::ChunkedPrefill, target::metal::Variant,
};

const MAX_CONTEXT: u32 = 1_024;
const PREFILL_CHUNK: u32 = 512;

struct ConstantEngine {
    embeddings: Tensor<f32>,
    prefill: Option<ChunkedPrefill<f32>>,
    paged: Option<PagedCacheSet>,
    paged_graph: Option<(BatchPhase, Graph<Tensor<f32>>)>,
    paged_query: Tensor<f32>,
    paged_key: Tensor<f32>,
    paged_value: Tensor<f32>,
}

#[export_engine]
impl Engine for ConstantEngine {
    fn describe() -> EngineInfo {
        EngineInfo {
            prompt_block_tokens: None,
            vocab: 4,
            max_context: MAX_CONTEXT,
            tap_layers: vec![0],
            router_layers: vec![],
            serving: Some(ServingInfo {
                page_tokens: 4,
                planes: ["cache", "value"]
                    .map(|label| CachePlane {
                        elements_per_token: 64,
                        alignment: 4,
                        label: label.to_owned(),
                    })
                    .into(),
                active_lanes: 2,
                max_context: 32,
                prefill_chunk_multiple: std::num::NonZeroU32::new(2),
                batch_shapes: [BatchSelection::Forward, BatchSelection::Masked]
                    .map(|selection| BatchShape {
                        phase: BatchPhase::Decode,
                        bucket: 2,
                        tokens: 1,
                        selection,
                    })
                    .into(),
            }),
        }
    }

    fn load(_weights: &Weights<'_>, config: EngineLoadConfig) -> Result<Self> {
        if !config.tunings.is_empty()
            && (config.tunings != ["fused"]
                || config.kv_cache != KvCacheFormat::F16
                || config.fixed_variant_picks.len() != 1
                || config.fixed_variant_picks[0].site != "dense.decode"
                || config.fixed_variant_picks[0].name != "matmul.gemv-transposed"
                || config.variant_rule_picks.len() != 1
                || config.variant_rule_picks[0].site != "attention.decode"
                || config.variant_rule_picks[0].parameter != "position"
                || config.variant_rule_picks[0].arms.len() != 1
                || config.variant_rule_picks[0].arms[0].lo != 0
                || config.variant_rule_picks[0].arms[0].hi != 1023
                || config.variant_rule_picks[0].arms[0].name != "sdpa.decomposed")
        {
            return Err(forja_sdk::Error::loading(
                "load selections did not round trip",
            ));
        }
        Ok(Self {
            embeddings: Tensor::from_slice(&[1.0; 16], &[4, 4])?,
            prefill: Some(ChunkedPrefill::new(MAX_CONTEXT, PREFILL_CHUNK)?),
            paged: None,
            paged_graph: None,
            paged_query: Tensor::zeros(&[1, 1, 64])?,
            paged_key: Tensor::zeros(&[1, 1, 64])?,
            paged_value: Tensor::zeros(&[1, 1, 64])?,
        })
    }

    fn load_paged(
        weights: &Weights<'_>,
        pool: Option<PagedPool<'_>>,
        config: EngineLoadConfig,
    ) -> Result<Self> {
        let prompt_tokens = config.max_context.unwrap_or(32).min(32);
        let mut engine = Self::load(weights, config)?;
        engine.paged = pool
            .map(|pool| {
                pool.cache_set_with_state(
                    1,
                    &[forja_sdk::PagedStateDescriptor {
                        dtype: forja_sdk::DType::F32,
                        shape: vec![1, 64],
                    }],
                    forja_sdk::PagedInputLayout {
                        vocab: 4,
                        prompt_tokens,
                    },
                )
            })
            .transpose()?;
        Ok(engine)
    }

    fn decode(&mut self, _input: DecodeInput) -> Result<DecodeOutput> {
        let (logits, token) = self.feedback_logits()?;
        self.paged
            .as_ref()
            .ok_or_else(|| forja_sdk::Error::loading("feedback slot is missing"))?
            .slot()
            .store_feedback(0, &token)?;
        forja_sdk::eval()?;
        Ok(DecodeOutput { logits, token })
    }

    fn begin_decode(&mut self, _input: BeginDecodeInput) -> Result<Tensor<f32>> {
        let (logits, _) = self.feedback_logits()?;
        forja_sdk::eval()?;
        Ok(logits)
    }

    fn batch_forward(&mut self, input: BatchInput<'_>) -> Result<Tensor<f32>> {
        let inputs = input
            .slot
            .tensors()?
            .inputs
            .ok_or_else(|| forja_sdk::Error::loading("batch inputs are missing"))?;
        let values = input
            .lanes
            .iter()
            .enumerate()
            .map(|(lane, active)| {
                active
                    .as_ref()
                    .map(|_| {
                        input
                            .slot
                            .token(
                                u32::try_from(lane)
                                    .map_err(|_| forja_sdk::Error::loading("lane exceeds u32"))?,
                            )
                            .and_then(|token| Self::token_value(&token))
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        for (lane, value) in values.into_iter().enumerate() {
            if let Some(value) = value {
                let lane = u32::try_from(lane)
                    .map_err(|_| forja_sdk::Error::loading("lane exceeds u32"))?;
                self.feedback_logits_value(lane, value)?
                    .reshape(&[1, 4])?
                    .copy_into(&mut inputs.logits.narrow(0, lane, 1)?)?;
            }
        }
        forja_sdk::eval()?;
        inputs.logits.narrow(0, 0, input.shape.bucket)
    }

    fn finish_decode(
        &mut self,
        _logits: &Tensor<f32>,
        start_pos: u32,
        _sampling: forja_sdk::SamplingParams,
        _allow: &[u32],
    ) -> Result<DecodeOutput> {
        let slot = self
            .paged
            .as_ref()
            .ok_or_else(|| forja_sdk::Error::loading("selection slot is missing"))?
            .slot();
        let inputs = slot
            .tensors()?
            .inputs
            .ok_or_else(|| forja_sdk::Error::loading("selection inputs are missing"))?;
        let logits = inputs.logits.narrow(0, 0, 1)?;
        let token = logits.sample_masked(
            &inputs.sampling.narrow(0, 0, 1)?.reshape(&[5])?,
            &inputs.allow.narrow(0, 0, 1)?.reshape(&[1])?,
            start_pos,
        )?;
        token.copy_into(&mut inputs.selected.narrow(0, 0, 1)?)?;
        slot.store_feedback(0, &token)?;
        forja_sdk::eval()?;
        Ok(DecodeOutput {
            logits: logits.reshape(&[4])?,
            token,
        })
    }

    fn rewind_decode(&mut self, _next_pos: u32) -> Result<()> {
        Ok(())
    }

    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
        if self.paged.is_some() {
            return self.paged_step(&input);
        }
        if input.tokens.shape()[0] > 1 {
            let mut prefill = self
                .prefill
                .take()
                .ok_or_else(|| forja_sdk::Error::loading("prefill state is unavailable"))?;
            let logits = prefill.replay(
                &input.tokens,
                input.start_pos,
                |tokens, _, last, _, _, _| {
                    self.embeddings
                        .embedding(tokens)?
                        .narrow(0, last, 1)?
                        .reshape(&[4])?
                        .contiguous()
                },
            );
            self.prefill = Some(prefill);
            return Ok(StepOutput {
                logits: logits?,
                taps: Vec::new(),
                router_logits: Vec::new(),
            });
        }
        let token = 1.0_f32;
        Ok(StepOutput {
            logits: Tensor::from_slice(&[token, 2.0, 3.0, 4.0], &[4])?,
            taps: input
                .taps
                .then(|| Tensor::from_slice(&[token, token], &[1, 2]))
                .transpose()?
                .into_iter()
                .collect(),
            router_logits: Vec::new(),
        })
    }
}

impl ConstantEngine {
    fn feedback_logits(&self) -> Result<(Tensor<f32>, Tensor<u32>)> {
        let set = self
            .paged
            .as_ref()
            .ok_or_else(|| forja_sdk::Error::loading("feedback smoke needs a serving slot"))?;
        let token = set.slot().token(0)?;
        let value = Self::token_value(&token)?;
        Ok((self.feedback_logits_value(0, value)?, token))
    }

    fn token_value(token: &Tensor<u32>) -> Result<f32> {
        let value = token
            .to_vec()?
            .first()
            .copied()
            .ok_or_else(|| forja_sdk::Error::loading("feedback token is missing"))?;
        Ok(f32::from(u16::try_from(value).map_err(|_| {
            forja_sdk::Error::loading("feedback token exceeds u16")
        })?))
    }

    fn feedback_logits_value(&self, lane: u32, value: f32) -> Result<Tensor<f32>> {
        let values = Tensor::from_slice(&[value; 64], &[1, 64])?;
        let weights = Tensor::from_slice(&[1.0; 128], &[2, 64])?;
        self.paged
            .as_ref()
            .ok_or_else(|| forja_sdk::Error::loading("feedback slot is missing"))?
            .state_at(lane, 0)?
            .causal_convolution(&values, &weights)?
            .reshape(&[64])?
            .narrow(0, 0, 4)
    }

    fn paged_step(&mut self, input: &StepInput) -> Result<StepOutput> {
        let token = self
            .paged
            .as_ref()
            .ok_or_else(|| forja_sdk::Error::loading("paged smoke cache is missing"))?
            .slot()
            .token(0)?
            .to_vec()?
            .first()
            .copied()
            .ok_or_else(|| forja_sdk::Error::loading("paged smoke token is missing"))?;
        let value = f32::from(
            u16::try_from(token)
                .map_err(|_| forja_sdk::Error::loading("paged smoke token exceeds u16"))?,
        );
        self.paged_value.write(&vec![value; 64])?;
        if self
            .paged_graph
            .as_ref()
            .is_none_or(|(phase, _)| *phase != input.phase)
        {
            let parameter = Param::new(0..=31)?;
            let start = Dim::from(parameter.at(input.start_pos));
            let end = start.clone().checked_add(1)?;
            let cache = self
                .paged
                .as_ref()
                .ok_or_else(|| forja_sdk::Error::loading("paged smoke cache is missing"))?
                .layer(0)?;
            let history = self
                .paged
                .as_ref()
                .ok_or_else(|| forja_sdk::Error::loading("state cache is missing"))?
                .state_at(0, 0)?;
            let weights = Tensor::from_slice(&[1.0; 128], &[2, 64])?;
            let variant = Variant::new("paged-sdpa.vector")?;
            let slot = self
                .paged
                .as_ref()
                .ok_or_else(|| forja_sdk::Error::loading("capture slot is missing"))?
                .slot();
            self.paged_graph = Some((
                input.phase,
                slot.capture(&[&parameter], input.phase, || {
                    let inputs = slot
                        .tensors()?
                        .inputs
                        .ok_or_else(|| forja_sdk::Error::loading("capture inputs are missing"))?;
                    slot.token(0)?
                        .copy_into(&mut inputs.selected.narrow(0, 0, 1)?)?;
                    let _history = history
                        .causal_convolution(&self.paged_value.reshape(&[1, 64])?, &weights)?;
                    cache.append(&self.paged_key, &self.paged_value, &start)?;
                    cache
                        .attention(&self.paged_query, 1.0, &start, &end, &variant)?
                        .reshape(&[64])?
                        .narrow(0, 0, 4)
                })?,
            ));
        }
        let (_, graph) = self
            .paged_graph
            .as_ref()
            .ok_or_else(|| forja_sdk::Error::loading("paged smoke graph is missing"))?;
        graph.replay(&[input.start_pos])?;
        Ok(StepOutput {
            logits: graph.result().alias()?,
            taps: Vec::new(),
            router_logits: Vec::new(),
        })
    }
}
