use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value};

use crate::{ChatEvent, Tool};

const MAX_EVENTS: usize = 8_192;
const MAX_ARGUMENT_BYTES: usize = 256 * 1024;

/// One validated function call produced by a model.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GeneratedToolCall {
    /// Declared function name.
    pub name: String,
    /// Canonical JSON object containing validated arguments.
    pub arguments: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum ToolDelta {
    Start(String),
    Arguments(String),
}

pub(crate) struct ToolOutput {
    tools: BTreeMap<String, Tool>,
    selected: Option<String>,
    phase: Phase,
    events: usize,
    input_argument_bytes: usize,
    output_argument_bytes: usize,
}

enum Phase {
    BeforeCall,
    Call(Box<Call>),
    Closed(GeneratedToolCall),
}

struct Call {
    tool: Tool,
    properties: Vec<String>,
    required: BTreeSet<String>,
    values: Map<String, Value>,
    current: Option<(String, String)>,
    last_index: Option<usize>,
}

impl ToolOutput {
    pub(crate) fn new(tools: &[Tool], selected: Option<String>) -> Self {
        Self {
            tools: tools
                .iter()
                .cloned()
                .map(|tool| (tool.function.name.clone(), tool))
                .collect(),
            selected,
            phase: Phase::BeforeCall,
            events: 0,
            input_argument_bytes: 0,
            output_argument_bytes: 0,
        }
    }

    pub(crate) fn apply(&mut self, events: Vec<ChatEvent>) -> Result<Vec<ToolDelta>, String> {
        self.events = self
            .events
            .checked_add(events.len())
            .filter(|&events| events <= MAX_EVENTS)
            .ok_or_else(|| "chat format event count exceeds the request limit".to_owned())?;
        let mut deltas = Vec::new();
        for event in events {
            match event {
                ChatEvent::TextDelta(_) => {
                    return Err("chat format emitted text after the tool trigger".to_owned());
                }
                ChatEvent::CallStart(name) => self.start(name, &mut deltas)?,
                ChatEvent::ArgumentDelta((name, fragment)) => {
                    self.argument(name, fragment, &mut deltas)?;
                }
                ChatEvent::CallEnd => self.end(&mut deltas)?,
            }
        }
        Ok(deltas)
    }

    pub(crate) fn finish(&self) -> Result<GeneratedToolCall, String> {
        match &self.phase {
            Phase::Closed(call) => Ok(call.clone()),
            Phase::BeforeCall | Phase::Call(_) => {
                Err("chat format did not close one tool call".to_owned())
            }
        }
    }

    fn start(&mut self, name: String, deltas: &mut Vec<ToolDelta>) -> Result<(), String> {
        if !matches!(self.phase, Phase::BeforeCall)
            || self
                .selected
                .as_ref()
                .is_some_and(|selected| selected != &name)
        {
            return Err("chat format emitted an unexpected tool call".to_owned());
        }
        let tool = self
            .tools
            .get(&name)
            .cloned()
            .ok_or_else(|| "chat format emitted an undeclared tool name".to_owned())?;
        let schema = resolved(&tool.function.parameters, &tool.function.parameters)?;
        let properties: Vec<String> = schema
            .get("properties")
            .and_then(Value::as_object)
            .map(|properties| properties.keys().cloned().collect())
            .unwrap_or_default();
        let required = schema
            .get("required")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect();
        self.phase = Phase::Call(Box::new(Call {
            tool,
            properties,
            required,
            values: Map::new(),
            current: None,
            last_index: None,
        }));
        deltas.push(ToolDelta::Start(name));
        Ok(())
    }

    fn argument(
        &mut self,
        name: String,
        fragment: String,
        deltas: &mut Vec<ToolDelta>,
    ) -> Result<(), String> {
        self.input_argument_bytes = self
            .input_argument_bytes
            .checked_add(fragment.len())
            .filter(|&bytes| bytes <= MAX_ARGUMENT_BYTES)
            .ok_or_else(|| "tool arguments exceed the request byte limit".to_owned())?;
        let completed = {
            let Phase::Call(call) = &mut self.phase else {
                return Err("chat format emitted arguments outside a tool call".to_owned());
            };
            if call
                .current
                .as_ref()
                .is_some_and(|(current, _)| current == &name)
            {
                if let Some((_, value)) = &mut call.current {
                    value.push_str(&fragment);
                }
                return Ok(());
            }
            let index = call
                .properties
                .iter()
                .position(|property| property == &name)
                .ok_or_else(|| "chat format emitted an unknown argument name".to_owned())?;
            if call.last_index.is_some_and(|last| index <= last) || call.values.contains_key(&name)
            {
                return Err("chat format emitted repeated or out-of-order arguments".to_owned());
            }
            let completed = call.finish_parameter()?;
            call.last_index = Some(index);
            call.current = Some((name, fragment));
            completed
        };
        if let Some(delta) = completed {
            self.push_arguments(delta, deltas)?;
        }
        Ok(())
    }

    fn end(&mut self, deltas: &mut Vec<ToolDelta>) -> Result<(), String> {
        let Phase::Call(mut call) = std::mem::replace(&mut self.phase, Phase::BeforeCall) else {
            return Err("chat format closed a missing tool call".to_owned());
        };
        if let Some(delta) = call.finish_parameter()? {
            self.push_arguments(delta, deltas)?;
        }
        if !call
            .required
            .iter()
            .all(|name| call.values.contains_key(name))
        {
            return Err("chat format omitted a required argument".to_owned());
        }
        let arguments = serde_json::to_string(&call.values)
            .map_err(|error| format!("cannot encode tool arguments: {error}"))?;
        let closing = if call.values.is_empty() { "{}" } else { "}" };
        self.push_arguments(closing.to_owned(), deltas)?;
        self.phase = Phase::Closed(GeneratedToolCall {
            name: call.tool.function.name,
            arguments,
        });
        Ok(())
    }

    fn push_arguments(
        &mut self,
        arguments: String,
        deltas: &mut Vec<ToolDelta>,
    ) -> Result<(), String> {
        self.output_argument_bytes = self
            .output_argument_bytes
            .checked_add(arguments.len())
            .filter(|&bytes| bytes <= MAX_ARGUMENT_BYTES)
            .ok_or_else(|| "canonical tool arguments exceed the request byte limit".to_owned())?;
        deltas.push(ToolDelta::Arguments(arguments));
        Ok(())
    }
}

impl Call {
    fn finish_parameter(&mut self) -> Result<Option<String>, String> {
        let Some((name, raw)) = self.current.take() else {
            return Ok(None);
        };
        let root = &self.tool.function.parameters;
        let schema = root
            .get("properties")
            .and_then(Value::as_object)
            .and_then(|properties| properties.get(&name))
            .ok_or_else(|| "tool argument schema is missing".to_owned())?;
        let value = parse_value(root, schema, &raw)?;
        let key = serde_json::to_string(&name)
            .map_err(|error| format!("cannot encode argument name: {error}"))?;
        let value_json = serde_json::to_string(&value)
            .map_err(|error| format!("cannot encode argument value: {error}"))?;
        let separator = if self.values.is_empty() { "{" } else { "," };
        self.values.insert(name, value);
        Ok(Some(format!("{separator}{key}:{value_json}")))
    }
}

fn parse_value(root: &Value, schema: &Value, raw: &str) -> Result<Value, String> {
    let schema = resolved(root, schema)?;
    let kind = schema.get("type").and_then(Value::as_str);
    let value = if kind == Some("string") {
        Value::String(raw.to_owned())
    } else {
        serde_json::from_str(raw).map_err(|_| "tool argument has the wrong JSON type".to_owned())?
    };
    validate_value(root, schema, &value)?;
    Ok(value)
}

fn validate_value(root: &Value, schema: &Value, value: &Value) -> Result<(), String> {
    let schema = resolved(root, schema)?;
    let kind = schema.get("type").and_then(Value::as_str);
    let valid_type = match kind {
        Some("string") => value.is_string(),
        Some("number") => value.is_number(),
        Some("integer") => value.as_i64().is_some() || value.as_u64().is_some(),
        Some("boolean") => value.is_boolean(),
        Some("array") => value.is_array(),
        Some("object") => value.is_object(),
        None => true,
        Some(_) => false,
    };
    if !valid_type
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| !values.contains(value))
    {
        return Err("tool argument violates its schema".to_owned());
    }
    if let Some(values) = value.as_array()
        && let Some(items) = schema.get("items")
    {
        for value in values {
            validate_value(root, items, value)?;
        }
    }
    if let Some(object) = value.as_object() {
        validate_object(root, schema, object)?;
    }
    Ok(())
}

fn validate_object(
    root: &Value,
    schema: &Value,
    object: &Map<String, Value>,
) -> Result<(), String> {
    let properties = schema.get("properties").and_then(Value::as_object);
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str);
    if !required.into_iter().all(|name| object.contains_key(name)) {
        return Err("tool argument object omits a required field".to_owned());
    }
    for (name, value) in object {
        let child = properties
            .and_then(|properties| properties.get(name))
            .ok_or_else(|| "tool argument object contains an unknown field".to_owned())?;
        validate_value(root, child, value)?;
    }
    Ok(())
}

fn resolved<'a>(root: &'a Value, mut schema: &'a Value) -> Result<&'a Value, String> {
    for _ in 0..=64 {
        let Some(reference) = schema.get("$ref").and_then(Value::as_str) else {
            return Ok(schema);
        };
        let pointer = reference
            .strip_prefix('#')
            .filter(|_| reference.starts_with("#/$defs/"))
            .ok_or_else(|| "tool argument schema reference is invalid".to_owned())?;
        schema = root
            .pointer(pointer)
            .ok_or_else(|| "tool argument schema reference is missing".to_owned())?;
    }
    Err("tool argument schema reference is recursive".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> Tool {
        serde_json::from_value(serde_json::json!({
            "type": "function",
            "function": {
                "name": "read_file",
                "description": null,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"type": "string", "enum": ["create", "append"]}
                    },
                    "required": ["path", "content", "mode"],
                    "additionalProperties": false
                }
            }
        }))
        .unwrap()
    }

    #[test]
    fn validates_order_and_builds_canonical_argument_deltas() {
        let mut output = ToolOutput::new(&[tool()], None);
        let deltas = output
            .apply(vec![
                ChatEvent::CallStart("read_file".to_owned()),
                ChatEvent::ArgumentDelta(("path".to_owned(), "a.rs".to_owned())),
                ChatEvent::ArgumentDelta(("content".to_owned(), "forged".to_owned())),
                ChatEvent::ArgumentDelta(("mode".to_owned(), "create".to_owned())),
                ChatEvent::CallEnd,
            ])
            .unwrap();
        assert_eq!(
            deltas,
            [
                ToolDelta::Start("read_file".to_owned()),
                ToolDelta::Arguments("{\"path\":\"a.rs\"".to_owned()),
                ToolDelta::Arguments(",\"content\":\"forged\"".to_owned()),
                ToolDelta::Arguments(",\"mode\":\"create\"".to_owned()),
                ToolDelta::Arguments("}".to_owned()),
            ]
        );
        assert_eq!(
            output.finish().unwrap().arguments,
            r#"{"path":"a.rs","content":"forged","mode":"create"}"#
        );
    }

    #[test]
    fn rejects_wrong_names_order_types_and_missing_end() {
        let mut wrong_name = ToolOutput::new(&[tool()], None);
        assert!(
            wrong_name
                .apply(vec![ChatEvent::CallStart("write".to_owned())])
                .is_err()
        );

        let mut out_of_order = ToolOutput::new(&[tool()], None);
        assert!(
            out_of_order
                .apply(vec![
                    ChatEvent::CallStart("read_file".to_owned()),
                    ChatEvent::ArgumentDelta(("content".to_owned(), "forged".to_owned())),
                    ChatEvent::ArgumentDelta(("path".to_owned(), "a.rs".to_owned())),
                ])
                .is_err()
        );

        let mut wrong_type = ToolOutput::new(&[tool()], None);
        assert!(
            wrong_type
                .apply(vec![
                    ChatEvent::CallStart("read_file".to_owned()),
                    ChatEvent::ArgumentDelta(("path".to_owned(), "a.rs".to_owned())),
                    ChatEvent::ArgumentDelta(("content".to_owned(), "forged".to_owned())),
                    ChatEvent::ArgumentDelta(("mode".to_owned(), "truncate".to_owned())),
                    ChatEvent::CallEnd,
                ])
                .is_err()
        );

        let mut missing_end = ToolOutput::new(&[tool()], None);
        missing_end
            .apply(vec![ChatEvent::CallStart("read_file".to_owned())])
            .unwrap();
        assert!(missing_end.finish().is_err());
    }

    #[test]
    fn rejects_malformed_sequences_and_aggregate_floods() {
        let mut selected = ToolOutput::new(&[tool()], Some("other".to_owned()));
        assert!(
            selected
                .apply(vec![ChatEvent::CallStart("read_file".to_owned())])
                .is_err()
        );

        let mut text = ToolOutput::new(&[tool()], None);
        assert!(
            text.apply(vec![ChatEvent::TextDelta("forged".to_owned())])
                .is_err()
        );

        let mut before_start = ToolOutput::new(&[tool()], None);
        assert!(
            before_start
                .apply(vec![ChatEvent::ArgumentDelta((
                    "path".to_owned(),
                    "a.rs".to_owned()
                ))])
                .is_err()
        );

        let mut duplicate = ToolOutput::new(&[tool()], None);
        assert!(
            duplicate
                .apply(vec![
                    ChatEvent::CallStart("read_file".to_owned()),
                    ChatEvent::CallStart("read_file".to_owned()),
                ])
                .is_err()
        );

        let mut events = ToolOutput::new(&[tool()], None);
        events
            .apply(vec![ChatEvent::CallStart("read_file".to_owned())])
            .unwrap();
        for _ in 1..MAX_EVENTS {
            events
                .apply(vec![ChatEvent::ArgumentDelta((
                    "path".to_owned(),
                    String::new(),
                ))])
                .unwrap();
        }
        assert!(events.apply(vec![ChatEvent::CallEnd]).is_err());

        let mut bytes = ToolOutput::new(&[tool()], None);
        assert!(
            bytes
                .apply(vec![
                    ChatEvent::CallStart("read_file".to_owned()),
                    ChatEvent::ArgumentDelta((
                        "path".to_owned(),
                        "x".repeat(MAX_ARGUMENT_BYTES + 1),
                    )),
                ])
                .is_err()
        );
    }
}
