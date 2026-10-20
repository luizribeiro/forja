/// Renders the generated host schema in its canonical checked-in form.
pub(crate) fn render() -> Result<String, String> {
    let schema = forja_config::host_schema()?;
    serde_json::to_string(&schema)
        .map(|schema| format!("{schema}\n"))
        .map_err(|error| format!("cannot serialize host schema: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_host_schema_has_not_drifted() {
        assert_eq!(
            render().unwrap(),
            include_str!("../../../nix/host.schema.json")
        );
    }
}
