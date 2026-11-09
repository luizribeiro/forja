use std::error::Error;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Request {
    pub(crate) id: u64,
    pub(crate) output: std::path::PathBuf,
}

impl Request {
    pub(crate) fn parse(line: &str, expected_id: u64) -> Result<Self, Box<dyn Error>> {
        let request: Self = serde_json::from_str(line)?;
        if request.id != expected_id || !request.output.is_absolute() || request.output.exists() {
            return Err("worker requires ordered requests and new absolute output paths".into());
        }
        Ok(request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn worker_protocol_refuses_replays_config_changes_and_existing_outputs() {
        let output = std::env::temp_dir().join(format!("forja-worker-{}", std::process::id()));
        let line = serde_json::json!({"id":1,"output":output}).to_string();
        assert!(Request::parse(&line, 1).is_ok());
        assert!(Request::parse(&line, 2).is_err());
        assert!(Request::parse(r#"{"id":1,"output":"relative.json"}"#, 1).is_err());
        assert!(Request::parse(r#"{"id":1,"output":"/unused","reps":4}"#, 1).is_err());
        std::fs::write(&output, b"record").unwrap();
        assert!(Request::parse(&line, 1).is_err());
        std::fs::remove_file(output).unwrap();
    }
}
