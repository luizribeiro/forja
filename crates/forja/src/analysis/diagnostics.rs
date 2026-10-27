pub(super) fn over_100(percentages: impl IntoIterator<Item = f64>) -> bool {
    percentages.into_iter().any(|percent| percent > 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentages_at_the_ceiling_are_not_above_it() {
        assert!(!over_100([0.0, 100.0]));
        assert!(over_100([0.0, 100.01]));
    }
}
