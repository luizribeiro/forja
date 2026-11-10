use serde_json::Value;

pub(super) fn validate_samples(row: &Value) -> Result<(), String> {
    let identity = &row["identity"];
    let concurrency = identity["concurrency"]
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or("missing concurrency")?;
    let reps = identity["reps"]
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or("missing repetitions")?;
    let output = identity["output_tokens"]
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or("missing requested output count")?;
    let prompt = identity["prompt_tokens"]
        .as_u64()
        .filter(|value| *value > 0)
        .ok_or("missing prompt count")?;
    let samples = row["samples"]
        .as_array()
        .filter(|samples| u64::try_from(samples.len()).ok() == Some(reps))
        .ok_or("incomplete completion samples")?;
    for (index, sample) in samples.iter().enumerate() {
        if sample["repetition"].as_u64() != u64::try_from(index).ok()
            || sample["decode_wall_seconds"]
                .as_f64()
                .is_none_or(|value| !value.is_finite() || value <= 0.0)
        {
            return Err("invalid completion sample timing or order".to_owned());
        }
        for (field, expected) in [
            ("prompt_tokens", Value::from(prompt)),
            ("output_tokens", Value::from(output)),
            ("finish_reasons", Value::from("length")),
        ] {
            if sample[field].as_array().is_none_or(|values| {
                u64::try_from(values.len()).ok() != Some(concurrency)
                    || values.iter().any(|value| *value != expected)
            }) {
                return Err(format!("degenerate completion sample: {field}"));
            }
        }
    }
    Ok(())
}
