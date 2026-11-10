use serde_json::Value;

pub(crate) fn assert_improvement_drafted(row: &Value) -> Result<(), String> {
    let minimum = row["minimum_speedup"]
        .as_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or("missing row speedup threshold")?;
    let rounds = row["rounds"]
        .as_u64()
        .ok_or("missing observed draft rounds")?;
    if minimum > 1.0 && rounds == 0 {
        return Err(format!(
            "improvement row {} never entered drafting",
            row["identity"]["name"]
        ));
    }
    Ok(())
}

pub(crate) fn valid_output(tokens: u64, finish: &str, minimum: u64, maximum: u64) -> bool {
    (minimum..=maximum).contains(&tokens)
        && (finish == "stop" || (finish == "length" && tokens == maximum))
}

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
    let minimum = identity["min_output_tokens"]
        .as_u64()
        .filter(|value| *value > 0 && *value <= output)
        .ok_or("missing or invalid minimum output count")?;
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
        for field in ["prompt_tokens", "output_tokens", "finish_reasons"] {
            if sample[field]
                .as_array()
                .is_none_or(|values| u64::try_from(values.len()).ok() != Some(concurrency))
            {
                return Err(format!("incomplete completion sample: {field}"));
            }
        }
        for request in 0..usize::try_from(concurrency).map_err(|error| error.to_string())? {
            if sample["prompt_tokens"][request].as_u64() != Some(prompt)
                || sample["output_tokens"][request]
                    .as_u64()
                    .is_none_or(|tokens| {
                        !valid_output(
                            tokens,
                            sample["finish_reasons"][request].as_str().unwrap_or(""),
                            minimum,
                            output,
                        )
                    })
            {
                return Err(format!(
                    "invalid completion sample: output_tokens={}, finish_reasons={}, prompt_tokens={}; requested={output}, minimum={minimum}",
                    sample["output_tokens"], sample["finish_reasons"], sample["prompt_tokens"]
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn improvement_claims_require_actual_verify_rounds() {
        let mut row =
            serde_json::json!({"identity":{"name":"copy"},"minimum_speedup":1.15,"rounds":0});
        assert!(super::assert_improvement_drafted(&row).is_err());
        row["minimum_speedup"] = 0.98.into();
        super::assert_improvement_drafted(&row).unwrap();
        row["minimum_speedup"] = 1.15.into();
        row["rounds"] = 1.into();
        super::assert_improvement_drafted(&row).unwrap();
        row["rounds"] = serde_json::Value::Null;
        assert!(super::assert_improvement_drafted(&row).is_err());
    }
}
