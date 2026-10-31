mod eager;
#[cfg(target_family = "wasm")]
mod paged;
use crate::{Config, Model};
use eager::EagerFeedback;
use forja_sdk::{
    DecodeInput, DecodeOutput, Engine, EngineInfo, EngineLoadConfig, Error, Result, StepInput,
    StepOutput, Weights,
};
enum Feedback {
    Eager(Box<EagerFeedback>),
    Paged {
        #[cfg(target_family = "wasm")]
        catalog: Box<paged::Catalog>,
    },
}

#[cfg(target_family = "wasm")]
impl Feedback {
    fn catalog(&mut self) -> Result<&mut paged::Catalog> {
        match self {
            Self::Paged { catalog } => Ok(catalog),
            Self::Eager(_) => Err(Error::loading("batch decode requires a serving pool")),
        }
    }
}

/// Text engine with private eager feedback or host-owned serving state.
pub struct Qwen38 {
    model: Model,
    feedback: Feedback,
}
impl Qwen38 {
    /// Returns the finite serving shapes; the eight-lane shapes execute eagerly.
    #[must_use]
    pub fn serving_shapes() -> Vec<forja_sdk::BatchShape> {
        use forja_sdk::{BatchPhase, BatchSelection, BatchShape};
        [1, 2, 4, 8]
            .into_iter()
            .flat_map(|bucket| {
                [
                    BatchSelection::Combined,
                    BatchSelection::Forward,
                    BatchSelection::Masked,
                ]
                .map(move |selection| BatchShape {
                    phase: BatchPhase::Decode,
                    bucket,
                    tokens: 1,
                    selection,
                })
            })
            .chain([64, 512, 2048].map(|tokens| BatchShape {
                phase: BatchPhase::Prompt,
                bucket: 1,
                tokens,
                selection: BatchSelection::Combined,
            }))
            .collect()
    }
    /// Loads an explicit architecture into a host-owned serving pool.
    ///
    /// # Errors
    /// Refuses invalid configuration, weights, selections or serving capacity.
    pub fn with_paged_config(
        weights: &Weights<'_>,
        load: &EngineLoadConfig,
        config: Config,
        pool: forja_sdk::PagedPool<'_>,
    ) -> Result<Self> {
        let model = Model::load_paged(weights, load, config, pool)?;
        #[cfg(target_family = "wasm")]
        let mut model = model;
        let feedback = Feedback::Paged {
            #[cfg(target_family = "wasm")]
            catalog: Box::new(paged::Catalog::new(&mut model, load)?),
        };
        Ok(Self { model, feedback })
    }

    /// Loads an explicit architecture for trusted native tests.
    ///
    /// # Errors
    /// Returns configuration, weight, state or selection allocation errors.
    pub fn with_config(
        weights: &Weights<'_>,
        load: &EngineLoadConfig,
        config: Config,
    ) -> Result<Self> {
        let model = Model::load(weights, load, config)?;
        let context = load
            .max_context
            .ok_or_else(|| Error::loading("profile context is missing"))?;
        Ok(Self {
            model,
            feedback: Feedback::Eager(Box::new(EagerFeedback::new(context, config.vocab)?)),
        })
    }
}
#[cfg_attr(feature = "component", forja_sdk::export_engine)]
impl Engine for Qwen38 {
    fn describe() -> EngineInfo {
        let c = Config::model();
        EngineInfo {
            prompt_block_tokens: std::num::NonZeroU32::new(crate::PREFILL_BLOCK_TOKENS),
            vocab: c.vocab,
            max_context: 262_144,
            tap_layers: (1..=c.layers).collect(),
            router_layers: Vec::new(),
            serving: None,
        }
    }
    fn load(weights: &Weights<'_>, config: EngineLoadConfig) -> Result<Self> {
        Self::with_config(weights, &config, Config::model())
    }
    fn load_paged(
        weights: &Weights<'_>,
        pool: Option<forja_sdk::PagedPool<'_>>,
        config: EngineLoadConfig,
    ) -> Result<Self> {
        match pool {
            Some(pool) => Self::with_paged_config(weights, &config, Config::model(), pool),
            None => Self::load(weights, config),
        }
    }
    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
        match &mut self.feedback {
            Feedback::Eager(feedback) => feedback.invalidate(),
            Feedback::Paged {
                #[cfg(target_family = "wasm")]
                catalog,
            } => {
                #[cfg(target_family = "wasm")]
                if !input.taps {
                    let rows = paged::row_count(Some(&input.tokens))?;
                    let shape = forja_sdk::BatchShape {
                        phase: input.phase,
                        bucket: 1,
                        tokens: rows,
                        selection: forja_sdk::BatchSelection::Forward,
                    };
                    if catalog.run(&mut self.model, shape, &[input.start_pos])? {
                        let logits = paged::batch_output(&self.model, 0, 1)?.logits;
                        return Ok(StepOutput {
                            logits: logits.reshape(&[logits.shape()[1]])?,
                            taps: Vec::new(),
                            router_logits: Vec::new(),
                        });
                    }
                }
                let output = self.model.forward_paged(
                    &input.tokens,
                    &forja_sdk::Dim::from(input.start_pos),
                    0,
                    input.phase,
                    input.taps,
                )?;
                forja_sdk::eval()?;
                return Ok(output);
            }
        }
        match input.phase {
            forja_sdk::BatchPhase::Prompt => {
                self.model
                    .forward(&input.tokens, input.start_pos, input.taps)
            }
            forja_sdk::BatchPhase::Decode => {
                self.model
                    .decode_token(&input.tokens, input.start_pos, input.taps)
            }
        }
    }
    #[cfg(target_family = "wasm")]
    fn batch_step(&mut self, input: forja_sdk::BatchInput<'_>) -> Result<forja_sdk::BatchOutput> {
        self.feedback
            .catalog()?
            .batch(&mut self.model, &input, true)
    }

    #[cfg(target_family = "wasm")]
    fn batch_forward(
        &mut self,
        input: forja_sdk::BatchInput<'_>,
    ) -> Result<forja_sdk::Tensor<f32>> {
        self.feedback
            .catalog()?
            .batch(&mut self.model, &input, false)
            .map(|output| output.logits)
    }

    #[cfg(target_family = "wasm")]
    fn batch_select(
        &mut self,
        input: forja_sdk::BatchSelectInput<'_>,
    ) -> Result<forja_sdk::BatchOutput> {
        self.feedback.catalog()?.select_batch(&self.model, &input)
    }

    #[cfg(target_family = "wasm")]
    fn begin_decode(
        &mut self,
        input: forja_sdk::BeginDecodeInput,
    ) -> Result<forja_sdk::Tensor<f32>> {
        let rows = paged::row_count(input.tokens.as_ref())?;
        let shape = forja_sdk::BatchShape {
            phase: input.phase,
            bucket: 1,
            tokens: rows,
            selection: forja_sdk::BatchSelection::Forward,
        };
        if !self
            .feedback
            .catalog()?
            .run(&mut self.model, shape, &[input.start_pos])?
        {
            drop(paged::forward(
                &mut self.model,
                rows,
                &input.start_pos.into(),
                0,
                input.phase,
            )?);
            forja_sdk::eval()?;
        }
        let logits = paged::batch_output(&self.model, 0, 1)?.logits;
        logits.reshape(&[logits.shape()[1]])
    }

    #[cfg(target_family = "wasm")]
    fn finish_decode(
        &mut self,
        _logits: &forja_sdk::Tensor<f32>,
        start_pos: u32,
        _sampling: forja_sdk::SamplingParams,
        _allow: &[u32],
    ) -> Result<DecodeOutput> {
        let shape = forja_sdk::BatchShape {
            phase: forja_sdk::BatchPhase::Decode,
            bucket: 1,
            tokens: 1,
            selection: forja_sdk::BatchSelection::Masked,
        };
        if !self.feedback.catalog()?.replay(shape, &[start_pos])? {
            drop(paged::select(&self.model, 0, &start_pos.into(), true)?);
            forja_sdk::eval()?;
        }
        let output = paged::batch_output(&self.model, 0, 1)?;
        Ok(DecodeOutput {
            logits: output.logits.reshape(&[output.logits.shape()[1]])?,
            token: output.tokens,
        })
    }

    #[cfg(target_family = "wasm")]
    fn rewind_decode(&mut self, _next_pos: u32) -> Result<()> {
        match self.feedback {
            Feedback::Paged { .. } => Ok(()),
            Feedback::Eager(_) => Err(Error::loading("rewind requires a serving pool")),
        }
    }

    fn decode(&mut self, input: DecodeInput) -> Result<DecodeOutput> {
        match &mut self.feedback {
            Feedback::Eager(feedback) => feedback.decode(&mut self.model, input),
            Feedback::Paged {
                #[cfg(target_family = "wasm")]
                catalog,
            } => {
                #[cfg(target_family = "wasm")]
                {
                    let rows = paged::row_count(input.tokens.as_ref())?;
                    let position = input
                        .start_pos
                        .checked_add(rows)
                        .and_then(|end| end.checked_sub(1))
                        .ok_or_else(|| Error::loading("decode position overflowed"))?;
                    let shape = forja_sdk::BatchShape {
                        phase: input.phase,
                        bucket: 1,
                        tokens: rows,
                        selection: forja_sdk::BatchSelection::Combined,
                    };
                    if !catalog.run(&mut self.model, shape, &[input.start_pos])? {
                        drop(paged::forward(
                            &mut self.model,
                            rows,
                            &input.start_pos.into(),
                            0,
                            input.phase,
                        )?);
                        drop(paged::select(&self.model, 0, &position.into(), true)?);
                        forja_sdk::eval()?;
                    }
                    let output = paged::batch_output(&self.model, 0, 1)?;
                    let (logits, token) = (output.logits, output.tokens);
                    Ok(DecodeOutput {
                        logits: logits.reshape(&[logits.shape()[1]])?,
                        token,
                    })
                }
                #[cfg(not(target_family = "wasm"))]
                {
                    Err(Error::loading("paged sampling requires a component host"))
                }
            }
        }
    }
}
