use forja_sdk::EngineLoadConfig;

pub(super) fn tiny_load() -> EngineLoadConfig {
    EngineLoadConfig {
        max_context: Some(64),
        kv_cache: forja_sdk::KvCacheFormat::F16,
        replay: false,
        ..EngineLoadConfig::default()
    }
}
