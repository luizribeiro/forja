#[derive(Clone, Copy)]
pub(crate) struct Stats {
    pub(crate) median: f64,
    pub(crate) low: f64,
    pub(crate) high: f64,
}

pub(crate) fn stats(values: impl Iterator<Item = f64>) -> Stats {
    let mut values = values.collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    let median = if values.len().is_multiple_of(2) {
        f64::midpoint(values[middle - 1], values[middle])
    } else {
        values[middle]
    };
    let rank = confidence_rank(values.len());
    Stats {
        median,
        low: values[rank],
        high: values[values.len() - rank - 1],
    }
}

fn confidence_rank(count: usize) -> usize {
    let Ok(count) = u32::try_from(count) else {
        return 0;
    };
    let exponent = i32::try_from(count).unwrap_or(i32::MAX);
    let mut probability = 0.5_f64.powi(exponent);
    let mut cumulative = probability;
    let mut rank = 0;
    for index in 0..count / 2 {
        probability *= f64::from(count - index) / f64::from(index + 1);
        if cumulative + probability > 0.025 {
            break;
        }
        cumulative += probability;
        rank = index + 1;
    }
    usize::try_from(rank).unwrap_or_default()
}

pub(crate) fn synthetic_tokens(count: usize, vocab: u32) -> Vec<u32> {
    let mut state = 0x4d59_5df4_d0f3_3173_u64;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            u32::try_from(state % u64::from(vocab)).unwrap_or_default()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn summarizes_a_median_with_order_statistic_bounds() {
        let summary = stats((1..=30).map(f64::from));
        assert!((summary.median - 15.5).abs() < f64::EPSILON);
        assert!((summary.low - 10.0).abs() < f64::EPSILON);
        assert!((summary.high - 21.0).abs() < f64::EPSILON);
    }

    #[test]
    fn synthetic_tokens_are_stable_and_in_vocabulary() {
        let tokens = synthetic_tokens(3, 151_936);
        assert_eq!(tokens, [131_128, 117_465, 70_310]);
        assert!(tokens.iter().all(|&token| token < 151_936));
    }
}
