//! Parameterized single-token graph replay.

#![forbid(unsafe_code)]

use crate::optimized::Optimized;
use forja_sdk::{
    Dim, Graph, Param, Result, StepOutput, Tensor,
    nn::blocks::{ChunkedPrefill, PrefillExecution},
};

pub(crate) struct Replay {
    enabled: bool,
    token: Tensor<u32>,
    decode: Option<Graph<Tensor<f32>>>,
    prefill: Option<ChunkedPrefill<f32>>,
}

impl Replay {
    pub(crate) fn new(context: u32, enabled: bool, chunked: bool) -> Result<Self> {
        let chunk = 512.min(1 << context.ilog2());
        Ok(Self {
            enabled,
            prefill: if chunked && chunk >= 16 {
                Some(ChunkedPrefill::new(context, chunk)?)
            } else {
                None
            },
            token: Tensor::zeros(&[1])?,
            decode: None,
        })
    }

    pub(crate) fn step(
        &mut self,
        engine: &mut Optimized,
        tokens: &Tensor<u32>,
        start: u32,
        taps: bool,
    ) -> Result<StepOutput> {
        let sequence = tokens.shape()[0];
        if !taps && sequence == 1 && self.enabled {
            self.token.write(&tokens.to_vec()?)?;
            if self.decode.is_none() {
                let parameter = Param::new(0..=engine.context - 1)?;
                let position = parameter.at(start);
                let end = (position.clone() + 1)?;
                self.decode = Some(forja_sdk::capture(&[&parameter], || {
                    Ok(engine
                        .forward(
                            &self.token,
                            1,
                            &position.into(),
                            &end,
                            &0.into(),
                            start,
                            Some(&parameter),
                            false,
                        )?
                        .logits)
                })?);
            }
            let graph = self
                .decode
                .as_ref()
                .ok_or_else(|| forja_sdk::Error::loading("decode graph unavailable"))?;
            graph.replay(&[start])?;
            return Ok(StepOutput {
                logits: graph.result().alias()?,
                taps: Vec::new(),
                router_logits: Vec::new(),
            });
        }
        if !taps
            && sequence > 1
            && let Some(prefill) = &mut self.prefill
        {
            let logits = if self.enabled {
                prefill.replay(tokens, start, |tokens, sequence, last, start, end, mode| {
                    let (pick_start, parameter) = match mode {
                        PrefillExecution::Capture(parameter) => (0, Some(parameter)),
                        PrefillExecution::Lazy(start) => (start, None),
                    };
                    Ok(engine
                        .forward(
                            tokens, sequence, start, end, last, pick_start, parameter, false,
                        )?
                        .logits)
                })?
            } else {
                prefill.lazy(
                    tokens,
                    start,
                    |tokens, sequence, last, start, end, pick_start| {
                        Ok(engine
                            .forward(tokens, sequence, start, end, last, pick_start, None, false)?
                            .logits)
                    },
                )?
            };
            return Ok(StepOutput {
                logits,
                taps: Vec::new(),
                router_logits: Vec::new(),
            });
        }
        engine.forward(
            tokens,
            sequence,
            &Dim::from(start),
            &Dim::from(start).checked_add(sequence)?,
            &Dim::from(sequence - 1),
            start,
            None,
            taps,
        )
    }
}
