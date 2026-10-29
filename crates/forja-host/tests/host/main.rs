//! Integration coverage for the component host.

#![allow(clippy::expect_used, clippy::unwrap_used)]

#[path = "../common/mod.rs"]
mod common;
#[path = "../engine_runner.rs"]
mod engine_runner;
#[path = "../olmoe_engine.rs"]
mod olmoe_engine;
#[path = "../prefill_replay.rs"]
mod prefill_replay;
#[path = "../qwen3_coder_engine.rs"]
mod qwen3_coder_engine;
#[path = "../qwen_engine.rs"]
mod qwen_engine;
#[path = "../qwen_weights.rs"]
mod qwen_weights;
#[path = "../smollm2.rs"]
mod smollm2;
#[path = "../tensor_smoke.rs"]
mod tensor_smoke;
#[path = "../toy_mlp.rs"]
mod toy_mlp;
