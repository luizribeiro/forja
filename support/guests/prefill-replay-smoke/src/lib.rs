//! Minimal engine that exposes chunked-prefill replay behavior to host integration tests.

use forja_sdk::{
    Dim, Engine, EngineInfo, EngineLoadConfig, Result, StepInput, StepOutput, Tensor, Weights,
    export_engine, nn::blocks::ChunkedPrefill,
};

const MAX_CONTEXT: u32 = 32_768;
const PREFILL_CHUNK: u32 = 2_048;
const TAIL_CHUNK: u32 = 16;
const VOCAB: u32 = 4;

struct PrefillReplayEngine {
    embeddings: Tensor<f32>,
    cache: Tensor<f32>,
    prefill: Option<ChunkedPrefill<f32>>,
    replay: bool,
    forward_calls: u32,
}

#[export_engine]
impl Engine for PrefillReplayEngine {
    fn describe() -> EngineInfo {
        EngineInfo {
            prompt_block_tokens: None,
            vocab: VOCAB,
            max_context: MAX_CONTEXT,
            tap_layers: vec![0],
            router_layers: vec![],
            serving: None,
        }
    }

    fn load(_weights: &Weights<'_>, config: EngineLoadConfig) -> Result<Self> {
        if !config.tunings.is_empty()
            || !config.fixed_variant_picks.is_empty()
            || !config.variant_rule_picks.is_empty()
        {
            return Err(forja_sdk::Error::loading(
                "prefill replay smoke does not accept selections",
            ));
        }
        Ok(Self {
            embeddings: Tensor::from_slice(
                &[
                    0.0, 10.0, 20.0, 30.0, 1.0, 11.0, 21.0, 31.0, 2.0, 12.0, 22.0, 32.0, 3.0, 13.0,
                    23.0, 33.0,
                ],
                &[VOCAB, VOCAB],
            )?,
            cache: Tensor::zeros(&[MAX_CONTEXT])?,
            prefill: Some(
                ChunkedPrefill::new(MAX_CONTEXT, PREFILL_CHUNK)?.with_tail_chunk(TAIL_CHUNK)?,
            ),
            replay: config.replay,
            forward_calls: 0,
        })
    }

    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
        let [sequence] = input
            .tokens
            .shape()
            .try_into()
            .map_err(|_| forja_sdk::Error::loading("tokens must have rank one"))?;
        let mut prefill = self
            .prefill
            .take()
            .ok_or_else(|| forja_sdk::Error::loading("prefill state is unavailable"))?;
        let result = if self.replay {
            prefill.replay(
                &input.tokens,
                input.start_pos,
                |tokens, bucket, last, start, _, _| {
                    forward(
                        &self.embeddings,
                        &self.cache,
                        &mut self.forward_calls,
                        tokens,
                        bucket,
                        last,
                        start,
                    )
                },
            )
        } else {
            prefill.lazy(
                &input.tokens,
                input.start_pos,
                |tokens, bucket, last, start, _, _| {
                    forward(
                        &self.embeddings,
                        &self.cache,
                        &mut self.forward_calls,
                        tokens,
                        bucket,
                        last,
                        start,
                    )
                },
            )
        };
        self.prefill = Some(prefill);
        let _ = result?;
        let taps = if input.taps {
            vec![
                self.cache
                    .narrow(0, input.start_pos, sequence)?
                    .reshape(&[sequence, 1])?,
            ]
        } else {
            Vec::new()
        };
        let forward_calls = u16::try_from(self.forward_calls)
            .map_err(|_| forja_sdk::Error::loading("forward count does not fit f32 exactly"))?;
        Ok(StepOutput {
            logits: Tensor::from_slice(&[f32::from(forward_calls), 0.0, 0.0, 0.0], &[VOCAB])?,
            taps,
            router_logits: Vec::new(),
        })
    }
}

fn forward(
    embeddings: &Tensor<f32>,
    cache: &Tensor<f32>,
    forward_calls: &mut u32,
    tokens: &Tensor<u32>,
    bucket: u32,
    last: &Dim,
    start: &Dim,
) -> Result<Tensor<f32>> {
    *forward_calls = forward_calls
        .checked_add(1)
        .ok_or_else(|| forja_sdk::Error::loading("forward count overflowed"))?;
    let values = embeddings.embedding(tokens)?;
    let projected = values.narrow(1, 0, 1)?.contiguous()?.reshape(&[bucket])?;
    projected.copy_into(&mut cache.narrow(0, start, bucket)?)?;
    values.narrow(0, last, 1)?.reshape(&[VOCAB])?.contiguous()
}
