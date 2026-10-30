//! Component coverage for the small independent hybrid checkpoint.
#![deny(unsafe_code)]
#![cfg_attr(
    target_family = "wasm",
    allow(unsafe_code, reason = "component export shim")
)]
use forja_sdk::{
    DecodeInput, DecodeOutput, Engine, EngineInfo, EngineLoadConfig, Result, StepInput, StepOutput,
    Weights, export_engine,
};

enum Tiny {
    Full(qwen38::Qwen38),
    Recurrent(qwen38::DeltaNet),
}

#[export_engine]
impl Engine for Tiny {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: 33,
            max_context: 64,
            tap_layers: (1..=4).collect(),
            router_layers: Vec::new(),
            prompt_block_tokens: std::num::NonZeroU32::new(64),
            serving: Some(forja_sdk::ServingInfo {
                page_tokens: 64,
                active_lanes: 8,
                max_context: 64,
                prefill_chunk_multiple: None,
                batch_shapes: Vec::new(),
                planes: ["key", "value"]
                    .into_iter()
                    .map(|label| forja_sdk::CachePlane {
                        elements_per_token: 32,
                        alignment: 256,
                        label: label.into(),
                    })
                    .collect(),
            }),
        }
    }
    fn load(weights: &Weights<'_>, load: EngineLoadConfig) -> Result<Self> {
        qwen38::Qwen38::with_config(weights, &load, qwen38::Config::tiny()).map(Self::Full)
    }
    fn load_paged(
        weights: &Weights<'_>,
        pool: Option<forja_sdk::PagedPool<'_>>,
        load: EngineLoadConfig,
    ) -> Result<Self> {
        let Some(pool) = pool else {
            return Self::load(weights, load);
        };
        let prompt_tokens = load.max_context.unwrap_or(64).min(64);
        let config = qwen38::Config::tiny();
        let cache = std::rc::Rc::new(pool.cache_set_with_state(
            1,
            &config.state_planes()?[..2],
            forja_sdk::PagedInputLayout {
                vocab: 33,
                prompt_tokens,
            },
        )?);
        qwen38::DeltaNet::load_paged(
            &weights.scoped("language_model.model.layers.0.linear_attn"),
            config,
            cache,
            0,
        )
        .map(Self::Recurrent)
    }
    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
        match self {
            Self::Full(engine) => engine.step(input),
            Self::Recurrent(layer) => {
                let [token]: [u32; 1] = input.tokens.to_vec()?.try_into().map_err(|_| {
                    forja_sdk::Error::loading("recurrent fixture requires one token")
                })?;
                let x = forja_sdk::Tensor::constant(
                    &[forja_sdk::bf16::from_f64(f64::from(token)); 64],
                    &[1, 64],
                )?;
                let logits = layer
                    .deltanet_layer(&x, None)?
                    .narrow(1, 0, 33)?
                    .to_dtype()?
                    .reshape(&[33])?;
                forja_sdk::eval()?;
                Ok(StepOutput {
                    logits,
                    taps: Vec::new(),
                    router_logits: Vec::new(),
                })
            }
        }
    }
    fn decode(&mut self, input: DecodeInput) -> Result<DecodeOutput> {
        match self {
            Self::Full(engine) => engine.decode(input),
            Self::Recurrent(_) => Err(forja_sdk::Error::loading("recurrent fixture uses step")),
        }
    }
}
