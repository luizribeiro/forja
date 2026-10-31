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
    Decoder(qwen38::DecoderLayer),
    Attention(qwen38::Attention),
}

#[export_engine]
impl Engine for Tiny {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: 33,
            max_context: 2048,
            tap_layers: (1..=4).collect(),
            router_layers: Vec::new(),
            prompt_block_tokens: std::num::NonZeroU32::new(64),
            serving: Some(forja_sdk::ServingInfo {
                page_tokens: 64,
                active_lanes: 8,
                max_context: 2048,
                prefill_chunk_multiple: None,
                batch_shapes: qwen38::Qwen38::serving_shapes(),
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
        if load.tunings.is_empty() {
            return qwen38::Qwen38::with_paged_config(weights, &load, qwen38::Config::tiny(), pool)
                .map(Self::Full);
        }
        if load.tunings.iter().any(|name| name == "attention-fixture") {
            let context = load.max_context.unwrap_or(64);
            return qwen38::Attention::load_paged(
                &weights.scoped("language_model.model.layers.3.self_attn"),
                qwen38::Config::tiny(),
                std::rc::Rc::new(pool.cache_set(1)?),
                0,
                forja_sdk::Tensor::constant(&(0..context).collect::<Vec<_>>(), &[context])?,
            )
            .map(Self::Attention);
        }
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
        if load.tunings.iter().any(|name| name == "decoder-fixture") {
            return qwen38::DecoderLayer::load_paged(
                &weights.scoped("language_model.model.layers.0"),
                config,
                0,
                cache,
                &forja_sdk::Tensor::constant(&(0..64).collect::<Vec<_>>(), &[64])?,
            )
            .map(Self::Decoder);
        }
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
            Self::Attention(layer) => {
                let values = input.tokens.to_vec()?;
                let rows = u32::try_from(values.len())
                    .map_err(|_| forja_sdk::Error::loading("fixture rows overflow"))?;
                let x = values
                    .into_iter()
                    .flat_map(|token| [forja_sdk::bf16::from_f64(f64::from(token)); 64])
                    .collect::<Vec<_>>();
                let output = layer.paged_layer(
                    &forja_sdk::Tensor::constant(&x, &[rows, 64])?,
                    &forja_sdk::Dim::from(input.start_pos),
                    input.phase == forja_sdk::BatchPhase::Prompt,
                    None,
                    &forja_sdk::target::metal::Variant::new("paged-sdpa.vector")?,
                )?;
                let logits = output
                    .narrow(0, rows - 1, 1)?
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
            Self::Recurrent(_) | Self::Decoder(_) => {
                let [token]: [u32; 1] = input.tokens.to_vec()?.try_into().map_err(|_| {
                    forja_sdk::Error::loading("recurrent fixture requires one token")
                })?;
                let x = forja_sdk::Tensor::constant(
                    &[forja_sdk::bf16::from_f64(f64::from(token)); 64],
                    &[1, 64],
                )?;
                let output = match self {
                    Self::Recurrent(layer) => layer.deltanet_layer(&x, None)?,
                    Self::Decoder(layer) => layer.paged_layer(
                        &x,
                        0,
                        &forja_sdk::Dim::from(input.start_pos),
                        input.phase,
                        [None; 3],
                    )?,
                    _ => return Err(forja_sdk::Error::loading("invalid fixture mixer")),
                };
                let logits = output.narrow(1, 0, 33)?.to_dtype()?.reshape(&[33])?;
                forja_sdk::eval()?;
                Ok(StepOutput {
                    logits,
                    taps: Vec::new(),
                    router_logits: Vec::new(),
                })
            }
        }
    }
    fn batch_step(&mut self, input: forja_sdk::BatchInput<'_>) -> Result<forja_sdk::BatchOutput> {
        match self {
            Self::Full(engine) => engine.batch_step(input),
            _ => Err(forja_sdk::Error::loading("fixture uses step")),
        }
    }
    fn batch_forward(
        &mut self,
        input: forja_sdk::BatchInput<'_>,
    ) -> Result<forja_sdk::Tensor<f32>> {
        match self {
            Self::Full(engine) => engine.batch_forward(input),
            _ => Err(forja_sdk::Error::loading("fixture uses step")),
        }
    }
    fn batch_select(
        &mut self,
        input: forja_sdk::BatchSelectInput<'_>,
    ) -> Result<forja_sdk::BatchOutput> {
        match self {
            Self::Full(engine) => engine.batch_select(input),
            _ => Err(forja_sdk::Error::loading("fixture uses step")),
        }
    }
    fn begin_decode(
        &mut self,
        input: forja_sdk::BeginDecodeInput,
    ) -> Result<forja_sdk::Tensor<f32>> {
        match self {
            Self::Full(engine) => engine.begin_decode(input),
            _ => Err(forja_sdk::Error::loading("fixture uses step")),
        }
    }
    fn finish_decode(
        &mut self,
        logits: &forja_sdk::Tensor<f32>,
        start_pos: u32,
        sampling: forja_sdk::SamplingParams,
        allow: &[u32],
    ) -> Result<DecodeOutput> {
        match self {
            Self::Full(engine) => engine.finish_decode(logits, start_pos, sampling, allow),
            _ => Err(forja_sdk::Error::loading("fixture uses step")),
        }
    }
    fn rewind_decode(&mut self, next_pos: u32) -> Result<()> {
        match self {
            Self::Full(engine) => engine.rewind_decode(next_pos),
            _ => Err(forja_sdk::Error::loading("fixture uses step")),
        }
    }
    fn decode(&mut self, input: DecodeInput) -> Result<DecodeOutput> {
        match self {
            Self::Full(engine) => engine.decode(input),
            Self::Recurrent(_) | Self::Decoder(_) | Self::Attention(_) => {
                Err(forja_sdk::Error::loading("recurrent fixture uses step"))
            }
        }
    }
}
