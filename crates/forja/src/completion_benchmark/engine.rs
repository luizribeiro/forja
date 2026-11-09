use std::error::Error;

use forja_config::{DevConfig, HostConfig, Profile};
use forja_core::Backend;
use forja_serve::{CompletionSession, ServingEngine};

use super::{Mode, Options};

pub(super) async fn load<B: Backend + Send + Sync + 'static>(
    backend: B,
    limits: forja_host::Limits,
    options: &Options,
    config: &DevConfig,
    profile: &Profile,
    identity: &serde_json::Value,
) -> Result<CompletionSession<ServingEngine<B>>, Box<dyn Error>> {
    let fields = [
        "engine_sha256",
        "profile_hash",
        "weights_sha256",
        "tokenizer_sha256",
        "chat_format_sha256",
    ];
    let hashes = fields.map(|field| {
        identity[field]
            .as_str()
            .map(str::to_owned)
            .ok_or("missing completion artifact hash")
    });
    let [engine, profile_hash, weights, tokenizer, chat] = hashes;
    let loaded = crate::serve::completion_engine(
        backend,
        limits,
        &host_config(options, config),
        profile.clone(),
        crate::engine::engine_load_config(&options.engine, &config.engine)?,
        [engine?, profile_hash?, weights?, tokenizer?, chat?],
        options.mode == Mode::Draft,
    )
    .await?;
    if !loaded.service.supports_continuous_batching() {
        return Err("completion benchmarks require continuous decode".into());
    }
    if options.mode == Mode::Draft && !loaded.service.profile_lookup_available() {
        return Err("draft mode requires profile lookup and admitted Verify shapes; check dispatches_per_list".into());
    }
    Ok(CompletionSession::new(loaded.service))
}

pub(super) fn host_config(options: &Options, config: &DevConfig) -> HostConfig {
    let mut host = HostConfig {
        backend: config.backend.clone(),
        limits: config.limits.clone(),
        ..HostConfig::default()
    };
    host.model.engine.clone_from(&options.engine);
    host.model.model.clone_from(&options.model);
    host.model.chat_format.clone_from(&options.chat_format);
    host.backend.metal.graph_replay =
        forja_config::Choice::Fixed(config.backend.metal.resolve().graph_replay);
    host.serving.prefix_reuse = options.cached;
    host
}
