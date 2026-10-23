//! Integration coverage for the guest SDK and native facade.

#![allow(clippy::expect_used, clippy::unwrap_used)]

#[path = "../kernel_native.rs"]
mod kernel_native;
#[path = "../kernel_ui.rs"]
mod kernel_ui;
#[path = "../load_derive.rs"]
mod load_derive;
#[path = "../native_cpu.rs"]
mod native_cpu;
#[path = "../native_load.rs"]
mod native_load;
#[path = "../native_metal.rs"]
mod native_metal;
#[path = "../qwen_moe.rs"]
mod qwen_moe;
#[cfg(feature = "native")]
#[path = "../support/mod.rs"]
mod support;
