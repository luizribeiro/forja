use serde_json::Value;

const TERMINALS: &str = "STRING: /[^\\r\\n<][^\\r<]*/\nINTEGER: /-?(0|[1-9][0-9]*)/\nNUMBER: /-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/\nJSON: /[^\\r<]+/\n";

/// Builds the Qwen tool-call grammar for a normalized request.
///
/// # Errors
/// Returns an error when the JSON or selected tool is invalid.
pub fn grammar(
    tools_json: &str,
    choice_json: &str,
    response_json: &str,
) -> Result<(String, String), String> {
    let tools: Value = serde_json::from_str(tools_json).map_err(|error| error.to_string())?;
    let choice: Value = serde_json::from_str(choice_json).map_err(|error| error.to_string())?;
    let response: Value = serde_json::from_str(response_json).map_err(|error| error.to_string())?;
    if !response.is_null() && response["type"] != "text" {
        return Err("response format must be text".to_owned());
    }
    let selected = choice.pointer("/function/name").and_then(Value::as_str);
    let tools = tools
        .as_array()
        .ok_or_else(|| "tools must be an array".to_owned())?;
    let mut tool_rules = Vec::new();
    let mut parameter_rules = Vec::new();
    for tool in tools {
        let function = &tool["function"];
        let name = function["name"]
            .as_str()
            .ok_or_else(|| "function name is missing".to_owned())?;
        if selected.is_some_and(|selected| selected != name) {
            continue;
        }
        let schema = function["parameters"]
            .as_object()
            .ok_or_else(|| "parameters must be an object schema".to_owned())?;
        let properties = schema.get("properties").and_then(Value::as_object);
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .collect();
        let index = tool_rules.len();
        let mut body = format!(
            "tool_{index}: {}",
            literal(&format!("<function={name}>\n"))?
        );
        if let Some(properties) = properties {
            for (parameter_index, (parameter, schema)) in properties.iter().enumerate() {
                let rule = format!("param_{index}_{parameter_index}");
                body.push_str(if required.contains(&parameter.as_str()) {
                    " "
                } else {
                    " ["
                });
                body.push_str(&rule);
                if !required.contains(&parameter.as_str()) {
                    body.push(']');
                }
                parameter_rules.push(format!(
                    "{rule}: {} {} {}",
                    literal(&format!("<parameter={parameter}>\n"))?,
                    value_rule(schema)?,
                    literal("\n</parameter>\n")?
                ));
            }
        }
        body.push(' ');
        body.push_str(&literal("</function>\n</tool_call>")?);
        tool_rules.push(body);
    }
    let starts = (0..tool_rules.len())
        .map(|index| format!("tool_{index}"))
        .collect::<Vec<_>>();
    if starts.is_empty() {
        return Err("tool choice selects no function".to_owned());
    }
    tool_rules.extend(parameter_rules);
    let grammar = format!(
        "start: {}\n{}\n{TERMINALS}",
        starts.join(" | "),
        tool_rules.join("\n")
    );
    Ok((grammar, "<tool_call>\n".to_owned()))
}

fn value_rule(schema: &Value) -> Result<String, String> {
    let kind = schema.get("type").and_then(Value::as_str);
    if let Some(values) = schema.get("enum") {
        let values = values
            .as_array()
            .filter(|values| !values.is_empty())
            .ok_or_else(|| "enum must be a nonempty array".to_owned())?;
        return values
            .iter()
            .map(|value| match kind {
                Some("string") => value
                    .as_str()
                    .ok_or_else(|| "string enum contains another type".to_owned())
                    .and_then(literal),
                Some("number") if value.is_number() => literal(&value.to_string()),
                Some("integer") if value.as_i64().is_some() || value.as_u64().is_some() => {
                    literal(&value.to_string())
                }
                Some("boolean") if value.is_boolean() => literal(&value.to_string()),
                _ => Err("enum type cannot be enforced".to_owned()),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|rules| rules.join(" | "));
    }
    Ok(match kind {
        Some("integer") => "INTEGER",
        Some("number") => "NUMBER",
        Some("boolean") => "\"true\" | \"false\"",
        Some("array" | "object") => "JSON",
        _ => "STRING",
    }
    .to_owned())
}

fn literal(value: &str) -> Result<String, String> {
    serde_json::to_string(value).map_err(|error| error.to_string())
}
