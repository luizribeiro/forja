use std::{error::Error, fs, path::Path};

use forja_config::read_profile;
use forja_host::{EngineLoadConfig, FixedVariantPick, VariantPickArm, VariantRulePick};

pub use forja_draft_testing as draft;

pub fn load_config(component: &Path) -> Result<EngineLoadConfig, Box<dyn Error>> {
    let profile = read_profile(&fs::read(component)?)?;
    Ok(EngineLoadConfig {
        max_context: Some(profile.workload().max_context.get()),
        replay: true,
        paged_cache: profile.serving().is_some_and(|serving| serving.paged_cache),
        kv_cache: match profile.numerics().kv_cache {
            forja_config::KvCacheFormat::F16 => forja_host::KvCacheFormat::F16,
            forja_config::KvCacheFormat::Bf16 => forja_host::KvCacheFormat::Bf16,
            forja_config::KvCacheFormat::F32 => forja_host::KvCacheFormat::F32,
        },
        tunings: profile.default_tunings().to_vec(),
        fixed_variant_picks: profile
            .variants()
            .fixed
            .iter()
            .map(|(site, name)| FixedVariantPick {
                site: site.clone(),
                name: name.clone(),
            })
            .collect(),
        variant_rule_picks: profile
            .variants()
            .rules
            .iter()
            .map(|(site, rule)| VariantRulePick {
                site: site.clone(),
                parameter: rule.parameter.clone(),
                arms: rule
                    .arms
                    .iter()
                    .map(|arm| VariantPickArm {
                        lo: arm.lo,
                        hi: arm.hi,
                        name: arm.name.clone(),
                    })
                    .collect(),
            })
            .collect(),
    })
}

/// Creates the protected raw-component sampling fixture without loading it.
pub async fn verify_runner() -> wasmtime::Result<forja_host::EngineRunner<forja_cpu::CpuBackend>> {
    let weights =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen38/model.safetensors");
    forja_host::EngineRunner::new(
        test_guests::verify_smoke(),
        forja_cpu::CpuBackend::new(),
        forja_host::Limits::new(1 << 24, 8, 1 << 20, 128, 1 << 20),
        weights,
    )
    .await
}
