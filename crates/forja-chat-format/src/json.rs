use minijinja::{Error, ErrorKind, Value, value::Kwargs};
use serde::Serialize;
use serde_json::ser::{Formatter, PrettyFormatter};
use std::io::{self, Write};

struct TemplateFormatter;

impl Formatter for TemplateFormatter {
    fn begin_array_value<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
    fn begin_object_key<W: ?Sized + Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> io::Result<()> {
        self.begin_array_value(writer, first)
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, writer: &mut W) -> io::Result<()> {
        writer.write_all(b": ")
    }
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "MiniJinja extracts keyword arguments as an owned value"
)]
pub(super) fn tojson(value: &Value, indent: Option<Value>, kwargs: Kwargs) -> Result<Value, Error> {
    let indent = indent.or(kwargs.get("indent")?);
    kwargs.assert_all_used()?;
    let indent = match indent {
        None => None,
        Some(value) => match bool::try_from(value.clone()).ok() {
            Some(true) => Some(2),
            Some(false) => None,
            None => Some(usize::try_from(value)?),
        },
    };
    if indent.is_some_and(|width| width > 128) {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "JSON indentation exceeds its limit",
        ));
    }
    let mut bytes = Vec::new();
    let result = if let Some(indent) = indent {
        let spaces = " ".repeat(indent);
        value.serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes,
            PrettyFormatter::with_indent(spaces.as_bytes()),
        ))
    } else {
        value.serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes,
            TemplateFormatter,
        ))
    };
    result.map_err(|error| {
        Error::new(
            ErrorKind::InvalidOperation,
            "cannot serialize template JSON",
        )
        .with_source(error)
    })?;
    String::from_utf8(bytes)
        .map(Value::from_safe_string)
        .map_err(|error| {
            Error::new(ErrorKind::InvalidOperation, "template JSON is not UTF-8").with_source(error)
        })
}

#[cfg(test)]
mod tests {
    use crate::ChatTemplate;

    #[test]
    fn json_preserves_definition_order_unicode_and_literal_xml_characters() {
        let template = ChatTemplate::compile("{{ data|tojson }}".to_owned()).unwrap();
        assert_eq!(
            template
                .render(
                    "[]",
                    "null",
                    r#"{"data":{"z":"<&'","a":{"city":"Montréal","x":[1,true]}}}"#
                )
                .unwrap(),
            r#"{"z": "<&'", "a": {"city": "Montréal", "x": [1, true]}}"#
        );
        let pretty = ChatTemplate::compile("{{ data|tojson(indent=4) }}".to_owned()).unwrap();
        assert_eq!(
            pretty.render("[]", "null", r#"{"data":{"x":1}}"#).unwrap(),
            "{\n    \"x\": 1\n}"
        );
    }
    #[test]
    fn json_refuses_unbounded_indentation() {
        let template = ChatTemplate::compile("{{ data|tojson(indent=129) }}".to_owned()).unwrap();
        assert!(template.render("[]", "null", r#"{"data":{}}"#).is_err());
    }
}
