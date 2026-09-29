use crate::{ConfigError, Layer, Origin, Schema, layer::validate};

/// Parses and validates one `--set KEY=VALUE` argument as a TOML line.
///
/// A bare-word value is interpreted as a string when it is not valid TOML syntax.
///
/// # Errors
///
/// Returns an error that identifies the numbered argument when the expression is not one TOML
/// line or does not match the schema.
pub fn set_layer<C: Schema>(index: usize, expression: &str) -> Result<Layer, ConfigError> {
    let context = format!("--set #{index} {expression}");
    if expression.contains(['\n', '\r']) {
        return Err(ConfigError::new(format!(
            "{context}: expected one TOML line"
        )));
    }
    let table = match toml::from_str(expression) {
        Ok(table) => table,
        Err(original) => parse_bare_word(expression)
            .map_err(|()| ConfigError::new(format!("{context}: {}", original.message())))?,
    };
    validate::<C>(&table).map_err(|message| ConfigError::new(format!("{context}: {message}")))?;
    Ok(Layer::new(Origin::Set(index), table))
}

fn parse_bare_word(expression: &str) -> Result<toml::Table, ()> {
    let assignment = assignment_index(expression).ok_or(())?;
    let (key, value) = expression.split_at(assignment);
    let value = value[1..].trim();
    if value.is_empty()
        || !value
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_-./~:".contains(character))
    {
        return Err(());
    }
    toml::from_str(&format!(
        "{} = {}",
        key.trim(),
        toml::Value::String(value.to_owned())
    ))
    .map_err(|_| ())
}

fn assignment_index(expression: &str) -> Option<usize> {
    let mut quote = None;
    let mut escaped = false;
    for (index, character) in expression.char_indices() {
        if escaped {
            escaped = false;
        } else if quote == Some('"') && character == '\\' {
            escaped = true;
        } else if matches!(character, '\'' | '"') {
            if quote == Some(character) {
                quote = None;
            } else if quote.is_none() {
                quote = Some(character);
            }
        } else if character == '=' && quote.is_none() {
            return Some(index);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use serde::{Deserialize, Serialize};

    use super::*;

    #[derive(Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(default, deny_unknown_fields)]
    struct SetConfig {
        labels: Labels,
    }

    #[derive(Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
    #[serde(default, deny_unknown_fields)]
    struct Labels {
        values: BTreeMap<String, String>,
    }

    #[test]
    fn parses_dotted_and_quoted_keys_with_a_bare_word() {
        let layer = set_layer::<SetConfig>(1, "labels.values.\"c.d\"=tier2").unwrap();
        let layered = crate::layer::<SetConfig>(vec![layer]).unwrap();
        assert_eq!(layered.config.labels.values["c.d"], "tier2");
    }

    #[test]
    fn validates_each_set_layer_with_its_number() {
        let error = set_layer::<crate::DevConfig>(2, "limits.tensor_ranks=5").unwrap_err();
        assert_eq!(
            error.to_string(),
            "--set #2 limits.tensor_ranks=5: unknown field `tensor_ranks`, expected one of `live_bytes`, `tensor_rank`, `tensor_elements`, `live_tensor_handles`, `live_kernels`, `live_graphs`, `read_bytes`, `guest_memory_bytes`, `table_elements`, `instances`, `dispatches_per_list`, `work_per_dispatch`, `guest_call_timeout`, `submission_timeout`, `gpu_time_budget`"
        );
    }
}
