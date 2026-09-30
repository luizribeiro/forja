//! Minimal engine component exported through the SDK macro.

use forja_sdk::{
    Engine, EngineInfo, EngineLoadConfig, Result, StepInput, StepOutput, Tensor, Weights,
    export_engine,
};

struct ConstantEngine;

#[export_engine]
impl Engine for ConstantEngine {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: 4,
            max_context: 33,
            tap_layers: vec![0],
            router_layers: vec![],
        }
    }

    fn load(_weights: &Weights<'_>, _config: EngineLoadConfig) -> Result<Self> {
        Ok(Self)
    }

    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
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
