use forja_config::{LogFormat, Logging};
use tracing_subscriber::{EnvFilter, fmt::format::FmtSpan};

/// Installs the configured process-wide structured logger.
pub(crate) fn init(config: &Logging) -> Result<(), String> {
    let filter = EnvFilter::try_new(&config.level)
        .map_err(|error| format!("invalid logging.level: {error}"))?;
    let spans = if config.include_spans {
        FmtSpan::NEW | FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };
    match config.format {
        LogFormat::Json => tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .with_span_events(spans)
            .with_target(false)
            .try_init()
            .map_err(|error| format!("cannot install logger: {error}"))?,
        LogFormat::Pretty => tracing_subscriber::fmt()
            .pretty()
            .with_env_filter(filter)
            .with_span_events(spans)
            .with_target(false)
            .try_init()
            .map_err(|error| format!("cannot install logger: {error}"))?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_filters_before_installing_a_subscriber() {
        let config = Logging {
            level: "[invalid".to_owned(),
            ..Logging::default()
        };
        assert!(init(&config).is_err());
    }
}
