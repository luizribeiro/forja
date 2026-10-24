use forja_config::{LogFormat, Logging};
use tracing_subscriber::{
    EnvFilter, Layer, Registry, fmt::format::FmtSpan, layer::SubscriberExt, reload,
    util::SubscriberInitExt,
};

type LoggingLayer = Box<dyn Layer<Registry> + Send + Sync>;

pub(crate) struct Controller {
    handle: reload::Handle<LoggingLayer, Registry>,
}

pub(crate) trait Reload {
    fn reconfigure(&self, config: &Logging) -> Result<(), String>;
}

impl Reload for Controller {
    fn reconfigure(&self, config: &Logging) -> Result<(), String> {
        self.handle
            .reload(layer(config)?)
            .map_err(|error| format!("cannot reload logger: {error}"))
    }
}

/// Installs the configured process-wide structured logger.
pub(crate) fn init(config: &Logging) -> Result<Controller, String> {
    let (layer, handle) = reload::Layer::new(layer(config)?);
    tracing_subscriber::registry()
        .with(layer)
        .try_init()
        .map_err(|error| format!("cannot install logger: {error}"))?;
    Ok(Controller { handle })
}

pub(crate) fn validate(config: &Logging) -> Result<(), String> {
    drop(layer(config)?);
    Ok(())
}

fn layer(config: &Logging) -> Result<LoggingLayer, String> {
    let filter = EnvFilter::try_new(&config.level)
        .map_err(|error| format!("invalid logging.level: {error}"))?;
    let spans = if config.include_spans {
        FmtSpan::NEW | FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };
    let layer = match config.format {
        LogFormat::Json => tracing_subscriber::fmt::layer()
            .json()
            .with_span_events(spans)
            .with_target(false)
            .with_filter(filter)
            .boxed(),
        LogFormat::Pretty => tracing_subscriber::fmt::layer()
            .pretty()
            .with_span_events(spans)
            .with_target(false)
            .with_filter(filter)
            .boxed(),
    };
    Ok(layer)
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
        assert!(validate(&config).is_err());
    }
}
