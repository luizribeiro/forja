use forja_core::BackendError;

pub(super) fn canonical(constants: &[(u32, u32)]) -> Result<Vec<(u32, u32)>, BackendError> {
    let mut constants = constants.to_vec();
    constants.sort_unstable();
    if constants.windows(2).any(|pair| pair[0].0 == pair[1].0) {
        return Err(BackendError::InvalidInput);
    }
    Ok(constants)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_indices_are_unique_and_order_independent() {
        assert_eq!(canonical(&[(2, 1), (0, 7)]), canonical(&[(0, 7), (2, 1)]));
        for values in [[(2, 1), (2, 1)], [(2, 1), (2, 7)]] {
            assert_eq!(canonical(&values), Err(BackendError::InvalidInput));
        }
        assert_eq!(canonical(&[]), Ok(Vec::new()));
    }
}
