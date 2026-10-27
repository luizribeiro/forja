use serde_json::Value;

pub(super) fn over_100(percentages: impl IntoIterator<Item = f64>) -> bool {
    percentages.into_iter().any(|percent| percent > 100.0)
}

pub(super) fn marker(row: &Value) -> &'static str {
    let flag = |name: &str| row[name].as_bool().unwrap_or(false);
    if flag("below_calibrated_floor") || flag("bound_exceeds_observed") {
        " [! outside calibration CI]"
    } else if flag("over_100_percent") {
        " [! >100%]"
    } else if flag("below_median_floor") || flag("median_bound_exceeds_observed") {
        " [! median bound exceeds time]"
    } else {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn visible_markers_distinguish_median_excess_from_ci_excess() {
        assert!(!over_100([0.0, 100.0]));
        assert!(over_100([0.0, 100.01]));
        assert_eq!(marker(&json!({"over_100_percent": true})), " [! >100%]");
        assert_eq!(
            marker(&json!({"over_100_percent": true, "below_calibrated_floor": true})),
            " [! outside calibration CI]"
        );
        assert_eq!(
            marker(&json!({"bound_exceeds_observed": true})),
            " [! outside calibration CI]"
        );
    }
}
