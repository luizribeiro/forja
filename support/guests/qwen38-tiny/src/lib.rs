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

struct Tiny(qwen38::Qwen38);

#[export_engine]
impl Engine for Tiny {
    fn describe() -> EngineInfo {
        EngineInfo {
            vocab: 33,
            max_context: 64,
            tap_layers: (1..=4).collect(),
            router_layers: Vec::new(),
            prompt_block_tokens: std::num::NonZeroU32::new(64),
            serving: None,
        }
    }
    fn load(weights: &Weights<'_>, load: EngineLoadConfig) -> Result<Self> {
        qwen38::Qwen38::with_config(weights, &load, qwen38::Config::tiny()).map(Self)
    }
    fn step(&mut self, input: StepInput) -> Result<StepOutput> {
        self.0.step(input)
    }
    fn decode(&mut self, input: DecodeInput) -> Result<DecodeOutput> {
        self.0.decode(input)
    }
}
