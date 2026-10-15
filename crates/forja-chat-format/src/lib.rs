//! Shared guest-side support for Hugging Face chat templates.
//!
//! The host supplies `now_unix_seconds` and `now_utc_offset_seconds` as render-option
//! globals. `strftime_now` formats that instant inside the import-free guest.

#![forbid(unsafe_code)]

use chrono::{DateTime, FixedOffset, Utc};
use minijinja::{AutoEscape, Environment, Error, ErrorKind, State, UndefinedBehavior};
use serde_json::Value as JsonValue;

/// A compiled embedded chat template.
pub struct ChatTemplate {
    environment: Environment<'static>,
}

impl ChatTemplate {
    /// Compiles an embedded Jinja template with Hugging Face-compatible behavior.
    ///
    /// # Errors
    ///
    /// Returns a template syntax error.
    pub fn compile(source: String) -> Result<Self, Error> {
        let mut environment = Environment::new();
        environment.set_auto_escape_callback(|_| AutoEscape::None);
        environment.set_undefined_behavior(UndefinedBehavior::Lenient);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function("raise_exception", raise_exception);
        environment.add_function("strftime_now", strftime_now);
        environment.add_template_owned("chat", source)?;
        Ok(Self { environment })
    }

    /// Renders JSON messages, tools, and options.
    ///
    /// Object members in `options_json` become template globals. `messages` and `tools` are always
    /// replaced by their corresponding interface arguments. The host-provided
    /// `now_unix_seconds` and `now_utc_offset_seconds` globals supply the clock for
    /// `strftime_now`.
    ///
    /// # Errors
    ///
    /// Returns a JSON, template lookup, or rendering error as guest-safe text.
    pub fn render(
        &self,
        messages_json: &str,
        tools_json: &str,
        options_json: &str,
    ) -> Result<String, String> {
        let messages = serde_json::from_str(messages_json).map_err(|error| error.to_string())?;
        let tools: JsonValue =
            serde_json::from_str(tools_json).map_err(|error| error.to_string())?;
        let options: JsonValue =
            serde_json::from_str(options_json).map_err(|error| error.to_string())?;
        let JsonValue::Object(mut context) = options else {
            return Err("chat options must be a JSON object".to_owned());
        };
        context.insert("messages".to_owned(), messages);
        if !tools.is_null() {
            context.insert("tools".to_owned(), tools);
        }
        self.environment
            .get_template("chat")
            .and_then(|template| template.render(JsonValue::Object(context)))
            .map_err(|error| error.to_string())
    }
}

fn raise_exception(message: String) -> Result<String, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message))
}

fn strftime_now(state: &State<'_, '_>, format: &str) -> Result<String, Error> {
    let seconds = state
        .lookup("now_unix_seconds")
        .and_then(|value| value.as_i64())
        .ok_or_else(|| clock_error("strftime_now has no Unix timestamp"))?;
    let offset_seconds = state
        .lookup("now_utc_offset_seconds")
        .and_then(|value| value.as_i64())
        .ok_or_else(|| clock_error("strftime_now has no UTC offset"))?;
    let offset = i32::try_from(offset_seconds)
        .ok()
        .and_then(FixedOffset::east_opt)
        .ok_or_else(|| clock_error("strftime_now has an invalid UTC offset"))?;
    let instant = DateTime::<Utc>::from_timestamp(seconds, 0)
        .ok_or_else(|| clock_error("strftime_now has an invalid Unix timestamp"))?;
    Ok(instant.with_timezone(&offset).format(format).to_string())
}

fn clock_error(message: &'static str) -> Error {
    Error::new(ErrorKind::InvalidOperation, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_lenient_undefined_values_and_python_methods() {
        let template = ChatTemplate::compile(
            "{{ missing }}{{ options.keys() | list | join(',') }}{{ messages[0].content }}"
                .to_owned(),
        )
        .unwrap();
        let output = template
            .render(
                r#"[{"content":"hello"}]"#,
                "null",
                r#"{"options":{"key":1}}"#,
            )
            .unwrap();
        assert_eq!(output, "keyhello");
    }

    #[test]
    fn installs_hugging_face_globals() {
        let globals = ChatTemplate::compile(
            "{{ 'yes' if raise_exception is defined and strftime_now is defined else 'no' }}"
                .to_owned(),
        )
        .unwrap();
        assert_eq!(globals.render("[]", "null", "{}").unwrap(), "yes");

        let date = ChatTemplate::compile("{{ strftime_now('%d %b %Y') }}".to_owned()).unwrap();
        assert_eq!(
            date.render(
                "[]",
                "null",
                r#"{"now_unix_seconds":1798761600,"now_utc_offset_seconds":0}"#,
            )
            .unwrap(),
            "01 Jan 2027"
        );

        let raises = ChatTemplate::compile("{{ raise_exception('boom') }}".to_owned()).unwrap();
        assert!(
            raises
                .render("[]", "null", "{}")
                .unwrap_err()
                .contains("boom")
        );
    }
}
