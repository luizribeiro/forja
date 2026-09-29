use std::{collections::BTreeMap, error::Error, fmt::Write as _};

use forja_config::{KeyPath, Origin};

use crate::args::ConfigShow;

pub(crate) fn render(options: &ConfigShow) -> Result<String, Box<dyn Error>> {
    let toml::Value::Table(mut table) = toml::Value::try_from(&options.layered.config)? else {
        return Err("configuration root must be a table".into());
    };
    if !options.defaults {
        table = without_defaults(table, "", &options.layered.origins);
    }
    let (value, origin_prefix) = select(toml::Value::Table(table), options.key.as_deref())?;
    if options.json {
        return render_json(
            value,
            options.origin,
            &origin_prefix,
            &options.layered.origins,
        );
    }
    let mut output = String::new();
    render_toml(
        &value,
        "",
        &origin_prefix,
        options.origin,
        &options.layered.origins,
        &mut output,
    );
    Ok(output)
}

fn without_defaults(
    table: toml::Table,
    prefix: &str,
    origins: &BTreeMap<KeyPath, Origin>,
) -> toml::Table {
    table
        .into_iter()
        .filter_map(|(key, value)| {
            let path = joined(prefix, &key);
            if let toml::Value::Table(table) = value {
                let table = without_defaults(table, &path, origins);
                (!table.is_empty()).then_some((key, toml::Value::Table(table)))
            } else {
                (origins.get(&KeyPath::new(path)) != Some(&Origin::Default)).then_some((key, value))
            }
        })
        .collect()
}

fn select(value: toml::Value, key: Option<&str>) -> Result<(toml::Value, String), Box<dyn Error>> {
    let Some(key) = key else {
        return Ok((value, String::new()));
    };
    if key.is_empty() {
        return Err("--key cannot be empty".into());
    }
    let mut selected = &value;
    for part in key.split('.') {
        selected = selected
            .as_table()
            .and_then(|table| table.get(part))
            .ok_or_else(|| format!("unknown configuration key {key:?}"))?;
    }
    Ok((selected.clone(), key.to_owned()))
}

fn render_json(
    value: toml::Value,
    show_origins: bool,
    prefix: &str,
    origins: &BTreeMap<KeyPath, Origin>,
) -> Result<String, Box<dyn Error>> {
    let rendered = if show_origins {
        let relevant = origins
            .iter()
            .filter(|(path, _)| {
                path.as_str() == prefix || prefix.is_empty() || {
                    path.as_str()
                        .strip_prefix(prefix)
                        .is_some_and(|suffix| suffix.starts_with('.'))
                }
            })
            .map(|(path, origin)| (path.as_str(), origin.to_string()))
            .collect::<BTreeMap<_, _>>();
        serde_json::json!({"config": value, "origins": relevant})
    } else {
        serde_json::to_value(value)?
    };
    Ok(format!("{}\n", serde_json::to_string_pretty(&rendered)?))
}

fn render_toml(
    value: &toml::Value,
    display_prefix: &str,
    origin_prefix: &str,
    show_origins: bool,
    origins: &BTreeMap<KeyPath, Origin>,
    output: &mut String,
) {
    let Some(table) = value.as_table() else {
        output.push_str(&value.to_string());
        append_origin(output, origin_prefix, value, show_origins, origins);
        output.push('\n');
        return;
    };
    for (key, value) in table.iter().filter(|(_, value)| !value.is_table()) {
        let _ = write!(output, "{} = {}", toml_key(key), value);
        append_origin(
            output,
            &joined(origin_prefix, key),
            value,
            show_origins,
            origins,
        );
        output.push('\n');
    }
    for (key, value) in table.iter().filter(|(_, value)| value.is_table()) {
        if !output.is_empty() && !output.ends_with("\n\n") {
            output.push('\n');
        }
        let display_path = joined(display_prefix, &toml_key(key));
        let _ = writeln!(output, "[{display_path}]");
        render_toml(
            value,
            &display_path,
            &joined(origin_prefix, key),
            show_origins,
            origins,
            output,
        );
    }
}

fn append_origin(
    output: &mut String,
    path: &str,
    value: &toml::Value,
    show: bool,
    origins: &BTreeMap<KeyPath, Origin>,
) {
    if show && path == "backend.metal.graph_replay" && value.as_str() == Some("auto") {
        output.push_str("  # unresolved: no Metal device");
    } else if show && let Some(origin) = origins.get(&KeyPath::new(path)) {
        let _ = write!(output, "  # {origin}");
    }
}

fn toml_key(key: &str) -> String {
    if key
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        key.to_owned()
    } else {
        toml::Value::String(key.to_owned()).to_string()
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
    use forja_config::{DevConfig, Layer, Origin, layer};

    use super::*;

    fn options(source: &str) -> ConfigShow {
        ConfigShow {
            layered: layer::<DevConfig>(vec![Layer::new(
                Origin::Set(1),
                toml::from_str(source).unwrap(),
            )])
            .unwrap(),
            origin: true,
            key: None,
            json: false,
            defaults: false,
        }
    }

    #[test]
    fn prints_non_default_values_with_origins() {
        assert_eq!(
            render(&options("[limits]\ntensor_rank = 7\n")).unwrap(),
            "[limits]\ntensor_rank = 7  # --set #1\n"
        );
    }

    #[test]
    fn selects_one_key_and_can_include_defaults() {
        let mut options = options("[limits]\ntensor_rank = 7\n");
        options.key = Some("limits.tensor_rank".to_owned());
        options.defaults = true;
        assert_eq!(render(&options).unwrap(), "7  # --set #1\n");
    }

    #[test]
    fn json_origins_are_machine_readable() {
        let mut options = options("[limits]\ntensor_rank = 7\n");
        options.json = true;
        let output: serde_json::Value = serde_json::from_str(&render(&options).unwrap()).unwrap();
        assert_eq!(output["config"]["limits"]["tensor_rank"], 7);
        assert_eq!(output["origins"]["limits.tensor_rank"], "--set #1");
    }

    #[test]
    fn leaves_auto_unresolved_without_a_metal_device() {
        let mut options = options("");
        options.key = Some("backend.metal.graph_replay".to_owned());
        options.defaults = true;
        assert_eq!(
            render(&options).unwrap(),
            "\"auto\"  # unresolved: no Metal device\n"
        );
    }
}
