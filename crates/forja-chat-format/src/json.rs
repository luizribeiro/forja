use minijinja::{Error, ErrorKind, Value, value::Kwargs};
use serde::Serialize;
use serde_json::ser::{Formatter, PrettyFormatter};
use std::io::{self, Write};

struct PythonNumbers<F>(F);

macro_rules! forward {
    ($($method:ident $(($argument:ident: $kind:ty))?),* $(,)?) => {
        $(fn $method<W: ?Sized + Write>(&mut self, writer: &mut W $(, $argument: $kind)?) -> io::Result<()> {
            self.0.$method(writer $(, $argument)?)
        })*
    };
}

impl<F: Formatter> Formatter for PythonNumbers<F> {
    forward!(
        begin_array,
        end_array,
        begin_array_value(first: bool),
        end_array_value,
        begin_object,
        end_object,
        begin_object_key(first: bool),
        end_object_key,
        begin_object_value,
        end_object_value
    );

    fn write_f32<W: ?Sized + Write>(&mut self, writer: &mut W, value: f32) -> io::Result<()> {
        self.write_f64(writer, f64::from(value))
    }
    fn write_f64<W: ?Sized + Write>(&mut self, writer: &mut W, value: f64) -> io::Result<()> {
        let scientific = format!("{value:e}");
        let (digits, exponent) = scientific
            .split_once('e')
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "non-finite JSON number"))?;
        let exponent: i32 = exponent
            .parse()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        if (-4..16).contains(&exponent) {
            let fixed = value.to_string();
            writer.write_all(fixed.as_bytes())?;
            if !fixed.contains('.') {
                writer.write_all(b".0")?;
            }
            Ok(())
        } else {
            write!(writer, "{digits}e{exponent:+03}")
        }
    }
}

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
        Some(value) if value.is_none() => None,
        Some(value) => match bool::try_from(value.clone()).ok() {
            Some(true) => Some(1),
            Some(false) => Some(0),
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
            PythonNumbers(PrettyFormatter::with_indent(spaces.as_bytes())),
        ))
    } else {
        value.serialize(&mut serde_json::Serializer::with_formatter(
            &mut bytes,
            PythonNumbers(TemplateFormatter),
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
    #[test]
    fn json_matches_python_float_notation_and_indent_controls() {
        let options = r#"{"data":[0.00001,0.0001,0.000001,1e16,1e20,-0.0,1.0,5e-324,1.7976931348623157e308]}"#;
        let expected =
            "[1e-05, 0.0001, 1e-06, 1e+16, 1e+20, -0.0, 1.0, 5e-324, 1.7976931348623157e+308]";
        for filter in ["tojson", "tojson(indent=none)"] {
            let template = ChatTemplate::compile(format!("{{{{ data|{filter} }}}}")).unwrap();
            assert_eq!(template.render("[]", "null", options).unwrap(), expected);
        }
        for (indent, spaces) in [("false", ""), ("true", " "), ("2", "  ")] {
            let template =
                ChatTemplate::compile(format!("{{{{ data|tojson(indent={indent}) }}}}")).unwrap();
            let pretty = format!(
                "[\n{spaces}{}\n]",
                expected[1..expected.len() - 1].replace(", ", &format!(",\n{spaces}"))
            );
            assert_eq!(template.render("[]", "null", options).unwrap(), pretty);
        }
    }
}
