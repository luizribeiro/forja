use schemars::generate::SchemaSettings;
use serde_json::Value;

use crate::{HOST_LEAVES, HostConfig};

/// Builds the complete host configuration JSON Schema.
///
/// # Errors
///
/// Returns an error if a registered host leaf is missing from the derived schema.
pub fn host_schema() -> Result<Value, String> {
    let mut settings = SchemaSettings::draft2020_12();
    settings.inline_subschemas = true;
    let mut schema = settings
        .into_generator()
        .into_root_schema_for::<HostConfig>()
        .to_value();
    for &(path, class) in HOST_LEAVES {
        let leaf = property_mut(&mut schema, path)
            .ok_or_else(|| format!("derived host schema is missing {path}"))?;
        let object = leaf
            .as_object_mut()
            .ok_or_else(|| format!("derived host schema leaf {path} is not an object"))?;
        object.insert(
            "x-forja-reload".to_owned(),
            Value::String(class.as_str().to_owned()),
        );
        if path.starts_with("model.") {
            object.insert("x-forja-absolute-path".to_owned(), Value::Bool(true));
        }
    }
    Ok(schema)
}

fn property_mut<'a>(schema: &'a mut Value, path: &str) -> Option<&'a mut Value> {
    let mut selected = schema;
    for part in path.split('.') {
        selected = selected.get_mut("properties")?.get_mut(part)?;
    }
    Some(selected)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_leaf_has_docs_a_default_and_reload_metadata() {
        let mut schema = host_schema().unwrap();
        for &(path, class) in HOST_LEAVES {
            let leaf = property_mut(&mut schema, path).unwrap();
            assert!(leaf.get("description").is_some(), "{path}");
            assert!(leaf.get("default").is_some(), "{path}");
            assert_eq!(leaf["x-forja-reload"], class.as_str(), "{path}");
            assert_eq!(
                leaf.get("x-forja-absolute-path").and_then(Value::as_bool),
                path.starts_with("model.").then_some(true),
                "{path}"
            );
        }
        assert_eq!(schema["additionalProperties"], false);
    }
}
