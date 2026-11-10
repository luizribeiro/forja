mod engine;
mod options;
mod record;
#[cfg(test)]
mod tests;
mod timing;
mod workload;

use forja_config::{BackendKind, DevConfig, Profile};
use golden_fixtures::sha256_file;
pub(crate) use options::{Mode, Options};
use serde_json::{Value, json};
use std::{error::Error, fs};
use tokenizers::Tokenizer;

fn prepare(
    options: &Options,
    config: &DevConfig,
) -> Result<(Profile, workload::Suite, Value), Box<dyn Error>> {
    let backend = match config.backend.kind {
        BackendKind::Cpu => crate::args::Backend::Cpu,
        BackendKind::Metal => crate::args::Backend::Metal,
    };
    let (profile, _) = crate::engine::validate_engine(
        &options.engine,
        &options.model,
        backend,
        &config.paths.scratch,
    )?;
    let suite = workload::Suite::load(&options.suite, &options.row, &options.group)?;
    if suite.target != profile.model().id {
        return Err("workload target differs from engine model".into());
    }
    let policy = serde_json::to_value(profile.prompt_lookup())?;
    for (key, expected) in suite
        .policy
        .as_object()
        .ok_or("suite policy must be an object")?
    {
        if policy.get(key.replace('_', "-")) != Some(expected) {
            return Err("workload and profile lookup policies differ".into());
        }
    }
    let identities = json!({"engine_sha256":sha256_file(&options.engine)?, "profile_hash":profile.sha256()?, "profile_family":profile.family(), "model":profile.model(), "numerics":profile.numerics(),
        "weights_sha256":profile.model().weights_sha256, "model_revision":profile.model().revision,
        "tokenizer_sha256":sha256_file(options.model.join("tokenizer.json"))?,
        "chat_format_sha256":sha256_file(&options.chat_format)?});
    Ok((profile, suite, identities))
}
