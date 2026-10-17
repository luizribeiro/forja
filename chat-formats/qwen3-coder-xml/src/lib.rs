//! Qwen XML tool calls used by Qwen3-Coder models.
//!
//! The format emits `<tool_call>`, `<function=...>`, and `<parameter=...>` tags. It currently has
//! no compatibility options; an empty options object is required so future model quirks remain
//! explicit and profile-controlled.

#![forbid(unsafe_code)]

use forja_chat_format::{Event, IncrementalParser, ToolFormat};
use serde_json::Value;

/// Qwen3-Coder's XML tool-call format.
pub struct Qwen3CoderXml;

/// Compatibility options for Qwen3-Coder XML.
#[derive(Default)]
pub struct Options;

impl ToolFormat for Qwen3CoderXml {
    type Options = Options;
    type Parser = Parser;

    fn options(options_json: &str) -> Result<Self::Options, String> {
        let options: Value =
            serde_json::from_str(options_json).map_err(|error| error.to_string())?;
        match options.as_object() {
            Some(options) if options.is_empty() => Ok(Options),
            Some(_) => Err("qwen3-coder-xml has no options".to_owned()),
            None => Err("tool format options must be an object".to_owned()),
        }
    }

    fn trigger(_: &Self::Options) -> &str {
        TRIGGER
    }

    fn normalize_messages(messages: &mut Value, _: &Self::Options) -> Result<(), String> {
        normalize_tool_calls(messages)
    }

    fn finish_render(
        mut rendered: String,
        request_options: &Value,
        _: &Self::Options,
    ) -> Result<String, String> {
        let choice = request_options.get("tool_choice");
        if choice.is_some_and(|choice| choice == "required" || choice.is_object()) {
            rendered.push_str(TRIGGER);
        }
        Ok(rendered)
    }

    fn grammar(
        tools_json: &str,
        choice_json: &str,
        response_json: &str,
        _: &Self::Options,
    ) -> Result<String, String> {
        grammar(tools_json, choice_json, response_json)
    }
}

const TERMINALS: &str = "STRING: /[^\\r\\n<][^\\r<]*/\nINTEGER: /-?(0|[1-9][0-9]*)/\nNUMBER: /-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?/\nJSON: /[^\\r<]+/\n";
const TRIGGER: &str = "<tool_call>\n";
const FUNCTION_START: &str = "<function=";
const PARAM_START: &str = "<parameter=";
const CALL_END: &str = "</function>\n</tool_call>";
const PARAM_END: &str = "\n</parameter>\n";

fn normalize_tool_calls(messages: &mut Value) -> Result<(), String> {
    let Some(messages) = messages.as_array_mut() else {
        return Err("messages must be an array".to_owned());
    };
    for message in messages {
        let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for call in calls {
            let Some(arguments) = call.pointer_mut("/function/arguments") else {
                continue;
            };
            if let Some(encoded) = arguments.as_str() {
                let decoded: Value =
                    serde_json::from_str(encoded).map_err(|error| error.to_string())?;
                if !decoded.is_object() {
                    return Err("tool arguments must encode an object".to_owned());
                }
                *arguments = decoded;
            }
        }
    }
    Ok(())
}

/// Incremental parser for one Qwen output stream.
#[derive(Default)]
pub struct Parser {
    buffer: String,
    activated: bool,
    done: bool,
    phase: ParsePhase,
}

#[derive(Default)]
enum ParsePhase {
    #[default]
    Function,
    Parameter,
    Value(String),
}

impl Parser {
    /// Parses the next ordered UTF-8 chunk.
    ///
    /// # Errors
    /// Returns an error for malformed XML or bytes following a completed call.
    pub fn parse(&mut self, chunk: &str) -> Result<Vec<Event>, String> {
        IncrementalParser::parse(self, chunk)
    }
}

impl IncrementalParser for Parser {
    fn parse(&mut self, chunk: &str) -> Result<Vec<Event>, String> {
        if self.done && !chunk.is_empty() {
            return Err("data follows tool call".to_owned());
        }
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        if !self.activated {
            if let Some(at) = self.buffer.find(TRIGGER) {
                if at > 0 {
                    events.push(Event::TextDelta(self.buffer.drain(..at).collect()));
                }
                self.buffer.drain(..TRIGGER.len());
                self.activated = true;
            } else {
                let held = (1..TRIGGER.len())
                    .rev()
                    .find(|&length| self.buffer.ends_with(&TRIGGER[..length]))
                    .unwrap_or(0);
                let ready = self.buffer.len() - held;
                if ready > 0 {
                    events.push(Event::TextDelta(self.buffer.drain(..ready).collect()));
                }
                return Ok(events);
            }
        }
        loop {
            match &mut self.phase {
                ParsePhase::Function => {
                    let Some(name) = take_tag(&mut self.buffer, FUNCTION_START)? else {
                        break;
                    };
                    events.push(Event::CallStart(name));
                    self.phase = ParsePhase::Parameter;
                }
                ParsePhase::Parameter => {
                    match consume_delimiter(&mut self.buffer, CALL_END) {
                        Some(true) => {
                            events.push(Event::CallEnd);
                            self.done = true;
                            if !self.buffer.is_empty() {
                                return Err("data follows tool call".to_owned());
                            }
                            break;
                        }
                        None => break,
                        Some(false) => {}
                    }
                    let Some(name) = take_tag(&mut self.buffer, PARAM_START)? else {
                        break;
                    };
                    self.phase = ParsePhase::Value(name);
                }
                ParsePhase::Value(name) => {
                    let Some(end) = self.buffer.find(PARAM_END) else {
                        break;
                    };
                    let value: String = self.buffer.drain(..end).collect();
                    self.buffer.drain(..PARAM_END.len());
                    events.push(Event::ArgumentDelta(name.clone(), value));
                    self.phase = ParsePhase::Parameter;
                }
            }
        }
        Ok(events)
    }
}

fn take_tag(buffer: &mut String, prefix: &str) -> Result<Option<String>, String> {
    if !prefix.starts_with(buffer.as_str()) && !buffer.starts_with(prefix) {
        return Err("invalid tool call tag".to_owned());
    }
    if !buffer.starts_with(prefix) {
        return Ok(None);
    }
    let Some(end) = buffer[prefix.len()..].find(">\n") else {
        return Ok(None);
    };
    let end = prefix.len() + end;
    let name = buffer[prefix.len()..end].to_owned();
    if name.is_empty()
        || name
            .bytes()
            .any(|byte| matches!(byte, b'<' | b'\r' | b'\n'))
    {
        return Err("invalid tool call name".to_owned());
    }
    buffer.drain(..end + 2);
    Ok(Some(name))
}

fn consume_delimiter(buffer: &mut String, delimiter: &str) -> Option<bool> {
    if buffer.starts_with(delimiter) {
        buffer.drain(..delimiter.len());
        return Some(true);
    }
    if delimiter.starts_with(buffer.as_str()) {
        return None;
    }
    Some(false)
}

/// Builds the Qwen tool-call grammar for a normalized request.
///
/// # Errors
/// Returns an error when the JSON or selected tool is invalid.
fn grammar(tools_json: &str, choice_json: &str, response_json: &str) -> Result<String, String> {
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
            let mut properties = properties.iter().collect::<Vec<_>>();
            properties.sort_unstable_by_key(|(parameter, _)| parameter.as_str());
            for (parameter_index, (parameter, schema)) in properties.into_iter().enumerate() {
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
    Ok(grammar)
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
