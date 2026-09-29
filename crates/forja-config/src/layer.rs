use std::{collections::BTreeMap, error::Error, fmt, path::PathBuf};

use serde::{Serialize, de::DeserializeOwned};

/// A dotted path to one scalar or array configuration value.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct KeyPath(String);

impl KeyPath {
    /// Returns the dotted path.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for KeyPath {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

/// The source of one configuration layer or leaf value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Origin {
    /// The typed schema default.
    Default,
    /// A supported environment variable.
    Env(&'static str),
    /// The developer's user configuration file.
    UserFile(PathBuf),
    /// An explicitly supplied configuration file.
    File(PathBuf),
    /// A command-line sugar flag.
    Flag(&'static str),
    /// A numbered `--set` argument.
    Set(usize),
    /// A benchmark vary point.
    Vary,
}

impl fmt::Display for Origin {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => formatter.write_str("default"),
            Self::Env(name) => write!(formatter, "{name}"),
            Self::UserFile(path) | Self::File(path) => path.display().fmt(formatter),
            Self::Flag(flag) => formatter.write_str(flag),
            Self::Set(index) => write!(formatter, "--set #{index}"),
            Self::Vary => formatter.write_str("vary point"),
        }
    }
}

/// One validated configuration fragment and its source.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    /// Source recorded for leaves in the fragment.
    pub origin: Origin,
    /// TOML table merged into the configuration root.
    pub table: toml::Table,
}

impl Layer {
    /// Creates a layer.
    #[must_use]
    pub const fn new(origin: Origin, table: toml::Table) -> Self {
        Self { origin, table }
    }
}

/// A typed configuration paired with the winning origin of every leaf.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Layered<C> {
    /// Fully merged and validated configuration.
    pub config: C,
    /// Winning source for every scalar or array leaf.
    pub origins: BTreeMap<KeyPath, Origin>,
}

/// A typed root that can be assembled from configuration layers.
pub trait Schema: Default + DeserializeOwned + Serialize {}

impl<T> Schema for T where T: Default + DeserializeOwned + Serialize {}

/// An invalid configuration layer or merged result.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigError(String);

impl ConfigError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Error for ConfigError {}

/// Merges validated layers over typed defaults and records each winning leaf source.
///
/// # Errors
///
/// Returns an error when defaults cannot be represented as a TOML table, a layer does not match
/// the schema, or the merged configuration is invalid.
pub fn layer<C: Schema>(layers: Vec<Layer>) -> Result<Layered<C>, ConfigError> {
    let mut table = schema_table(&C::default())?;
    let mut origins = BTreeMap::new();
    record_origins(&table, "", &Origin::Default, &mut origins);
    for next in layers {
        validate::<C>(&next.table)
            .map_err(|message| ConfigError::new(format!("{}: {message}", next.origin)))?;
        merge_table(&mut table, next.table, "", &next.origin, &mut origins);
    }
    let config = toml::Value::Table(table)
        .try_into()
        .map_err(|error| ConfigError::new(format!("merged configuration: {error}")))?;
    Ok(Layered { config, origins })
}

fn schema_table<C: Serialize>(config: &C) -> Result<toml::Table, ConfigError> {
    toml::Value::try_from(config)
        .map_err(|error| ConfigError::new(format!("cannot serialize defaults: {error}")))?
        .as_table()
        .cloned()
        .ok_or_else(|| ConfigError::new("configuration root must be a table"))
}

pub(crate) fn validate<C: Schema>(table: &toml::Table) -> Result<(), String> {
    toml::Value::Table(table.clone())
        .try_into::<C>()
        .map(|_| ())
        .map_err(|error: toml::de::Error| error.message().to_owned())
}

fn merge_table(
    base: &mut toml::Table,
    next: toml::Table,
    prefix: &str,
    origin: &Origin,
    origins: &mut BTreeMap<KeyPath, Origin>,
) {
    for (key, value) in next {
        let path = joined(prefix, &key);
        if let (Some(toml::Value::Table(base)), toml::Value::Table(next)) =
            (base.get_mut(&key), &value)
        {
            merge_table(base, next.clone(), &path, origin, origins);
        } else {
            record_value_origins(&value, &path, origin, origins);
            base.insert(key, value);
        }
    }
}

fn record_origins(
    table: &toml::Table,
    prefix: &str,
    origin: &Origin,
    origins: &mut BTreeMap<KeyPath, Origin>,
) {
    for (key, value) in table {
        record_value_origins(value, &joined(prefix, key), origin, origins);
    }
}

fn record_value_origins(
    value: &toml::Value,
    path: &str,
    origin: &Origin,
    origins: &mut BTreeMap<KeyPath, Origin>,
) {
    if let toml::Value::Table(table) = value {
        record_origins(table, path, origin, origins);
    } else {
        origins.insert(KeyPath(path.to_owned()), origin.clone());
    }
}

fn joined(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_owned()
    } else {
        format!("{prefix}.{key}")
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(default, deny_unknown_fields)]
    struct TestConfig {
        limits: TestLimits,
    }

    #[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(default, deny_unknown_fields)]
    struct TestLimits {
        count: u64,
        names: Vec<String>,
    }

    impl Default for TestLimits {
        fn default() -> Self {
            Self {
                count: 7,
                names: vec!["default".to_owned()],
            }
        }
    }

    #[test]
    fn merges_tables_and_replaces_leaf_values() {
        let first = toml::from_str("[limits]\ncount = 33\n").unwrap();
        let second = toml::from_str("[limits]\nnames = [\"one\", \"two\"]\n").unwrap();
        let layered = layer::<TestConfig>(vec![
            Layer::new(Origin::File("first.toml".into()), first),
            Layer::new(Origin::Set(1), second),
        ])
        .unwrap();
        assert_eq!(layered.config.limits.count, 33);
        assert_eq!(layered.config.limits.names, ["one", "two"]);
        assert_eq!(
            layered.origins.get(&KeyPath("limits.count".to_owned())),
            Some(&Origin::File("first.toml".into()))
        );
        assert_eq!(
            layered.origins.get(&KeyPath("limits.names".to_owned())),
            Some(&Origin::Set(1))
        );
    }
}
