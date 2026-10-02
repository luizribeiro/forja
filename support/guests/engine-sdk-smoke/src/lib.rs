//! Minimal engine component exported through the SDK macro.

use forja_sdk::{
    Engine, EngineInfo, EngineLoadConfig, Result, StepInput, StepOutput, Tensor, Weights,
    export_engine, nn::blocks::ChunkedPrefill,
};

const MAX_CONTEXT: u32 = 1_024;
const PREFILL_CHUNK: u32 = 512;

struct ConstantEngine {
    embeddings: Tensor<f32>,
    prefill: Option<ChunkedPrefill<f32>>,
}

#[export_engine]
impl Engine for ConstantEngine {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: 4,
            max_context: MAX_CONTEXT,
            tap_layers: vec![0],
            router_layers: vec![],
        }
    }

    fn load(_weights: &Weights<'_>, config: EngineLoadConfig) -> Result<Self> {
        if !config.tunings.is_empty()
            && (config.tunings != ["fused"]
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
        })
    }

    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
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
