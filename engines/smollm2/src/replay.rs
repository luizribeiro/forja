//! Parameterized single-token graph replay.

#![forbid(unsafe_code)]

use crate::optimized::Optimized;
use forja_sdk::{Dim, Graph, Param, Result, StepOutput, Tensor};

pub(crate) struct Replay {
    enabled: bool,
    token: Tensor<u32>,
    decode: Option<Graph<Tensor<f32>>>,
}

impl Replay {
    pub(crate) fn new(enabled: bool) -> Result<Self> {
        Ok(Self {
            enabled,
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
