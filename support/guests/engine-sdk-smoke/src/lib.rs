//! Minimal engine component exported through the SDK macro.

use forja_sdk::{
    CachePlane, Dim, Engine, EngineInfo, EngineLoadConfig, Graph, KvCacheFormat, PagedCacheSet,
    PagedPool, Param, Result, ServingInfo, StepInput, StepOutput, Tensor, Weights, capture,
    export_engine, nn::blocks::ChunkedPrefill, target::metal::Variant,
};

const MAX_CONTEXT: u32 = 1_024;
const PREFILL_CHUNK: u32 = 512;

struct ConstantEngine {
    embeddings: Tensor<f32>,
    prefill: Option<ChunkedPrefill<f32>>,
    paged: Option<PagedCacheSet>,
    paged_graph: Option<Graph<Tensor<f32>>>,
    paged_query: Tensor<f32>,
    paged_key: Tensor<f32>,
    paged_value: Tensor<f32>,
}

#[export_engine]
impl Engine for ConstantEngine {
    fn describe() -> EngineInfo {
        EngineInfo {
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
                batch_shapes: vec![],
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
        let mut engine = Self::load(weights, config)?;
        engine.paged = pool.map(|pool| pool.cache_set(1)).transpose()?;
        Ok(engine)
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
    fn paged_step(&mut self, input: &StepInput) -> Result<StepOutput> {
        let token = input
            .tokens
            .to_vec()?
            .first()
            .copied()
            .ok_or_else(|| forja_sdk::Error::loading("paged smoke token is missing"))?;
        let value = f32::from(
            u16::try_from(token)
                .map_err(|_| forja_sdk::Error::loading("paged smoke token exceeds u16"))?,
        );
        self.paged_value.write(&vec![value; 64])?;
        if self.paged_graph.is_none() {
            let parameter = Param::new(0..=31)?;
            let start = Dim::from(parameter.at(input.start_pos));
            let end = start.clone().checked_add(1)?;
            let cache = self
                .paged
                .as_ref()
                .ok_or_else(|| forja_sdk::Error::loading("paged smoke cache is missing"))?
                .layer(0)?;
            let variant = Variant::new("paged-sdpa.vector")?;
            self.paged_graph = Some(capture(&[&parameter], || {
                cache.append(&self.paged_key, &self.paged_value, &start)?;
                cache
                    .attention(&self.paged_query, 1.0, &start, &end, &variant)?
                    .reshape(&[64])?
                    .narrow(0, 0, 4)
            })?);
        }
        let graph = self
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
