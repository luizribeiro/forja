#[cfg(target_family = "wasm")]
use super::Model;
use crate::Qwen38;
#[cfg(target_family = "wasm")]
use forja_sdk::Tensor;
use forja_sdk::{Dim, Error, Result};

const PAIR_MAX_ROWS: u32 = Qwen38::VERIFY_ROWS;
const ALLOWED_BATCHES: [u32; Qwen38::DECODE_BUCKETS.len() + 1] = [
    Qwen38::DECODE_BUCKETS[0],
    Qwen38::DECODE_BUCKETS[1],
    Qwen38::VERIFY_ROWS,
    Qwen38::DECODE_BUCKETS[2],
    Qwen38::DECODE_BUCKETS[3],
];
const _: () = {
    assert!(Qwen38::VERIFY_ROWS == Qwen38::DECODE_BUCKETS[1] + 1);
    assert!(PAIR_MAX_ROWS < Qwen38::DECODE_BUCKETS[2]);
};

#[cfg(target_family = "wasm")]
impl Model {
    pub(crate) fn forward_decode_rows(
        &mut self,
        tokens: &Tensor<u32>,
        starts: &[Dim],
        lanes: &[u32],
    ) -> Result<Tensor<f32>> {
        let batch = validate_rows(tokens.shape(), starts, lanes, self.context)?;
        let paged = self
            .paged
            .as_ref()
            .ok_or_else(|| Error::loading("model has no serving slot"))?;
        let positions = Tensor::<u32>::zeros(&[batch])?;
        for (row, start) in starts.iter().enumerate() {
            let row =
                u32::try_from(row).map_err(|_| Error::loading("decode row index exceeds u32"))?;
            paged
                .positions
                .narrow(0, start.clone(), 1)?
                .copy_into(&mut positions.narrow(0, row, 1)?)?;
        }
        let quant =
            if batch == 1 {
                self.quant_decode.as_ref()
            } else {
                self.quant_decode_batch.as_ref().map(|[pair, wide]| {
                    if batch <= PAIR_MAX_ROWS { pair } else { wide }
                })
            };
        forja_sdk::profile_scope("embedding", "global", None)?;
        let mut hidden = self.embedding.embed(tokens)?;
        for layer in &mut self.layers {
            hidden = layer.decode_rows(&hidden, &positions, starts, lanes, quant, &paged.decode)?;
        }
        forja_sdk::profile_scope("output", "global", None)?;
        hidden = hidden.rms_norm(&self.norm, 1.0e-6)?;
        self.output.forward(&hidden, quant)?.to_dtype::<f32>()
    }
}

fn validate_rows(shape: &[u32], starts: &[Dim], lanes: &[u32], context: u32) -> Result<u32> {
    let batch =
        u32::try_from(starts.len()).map_err(|_| Error::loading("decode row count exceeds u32"))?;
    if !ALLOWED_BATCHES.contains(&batch)
        || lanes.len() != starts.len()
        || shape != [batch]
        || starts.iter().any(|start| start.value() >= context)
        || lanes.iter().any(|lane| *lane >= Qwen38::MAX_DECODE_ROWS)
    {
        return Err(Error::loading("invalid decode row span"));
    }
    for (lanes, starts) in lanes.windows(2).zip(starts.windows(2)) {
        if lanes[0] > lanes[1]
            || (lanes[0] == lanes[1] && starts[0].clone().checked_add(1)? != starts[1])
        {
            return Err(Error::loading(
                "decode lane rows must advance consecutively",
            ));
        }
    }
    Ok(batch)
}

#[cfg(all(test, not(target_family = "wasm")))]
mod tests {
    use super::*;

    #[test]
    fn repeated_lanes_require_consecutive_positions() {
        let check = |positions: &[u32], lanes: &[u32]| {
            let starts = positions.iter().copied().map(Dim::from).collect::<Vec<_>>();
            validate_rows(
                &[u32::try_from(starts.len()).unwrap()],
                &starts,
                lanes,
                8192,
            )
        };
        assert!(check(&[4097, 4098], &[7, 7]).is_ok());
        assert_eq!(check(&[4097, 4098, 4099], &[7, 7, 7]).unwrap(), 3);
        assert!(check(&[4097, 7, 8], &[0, 7, 7]).is_ok());
        for positions in [[7, 7, 8], [7, 9, 10], [9, 8, 7]] {
            assert!(check(&positions, &[7, 7, 7]).is_err());
        }
        assert!(check(&[7, 8], &[7, 0]).is_err());
        assert!(check(&[8190, 8191, 8192], &[7, 7, 7]).is_err());
        assert!(check(&[7, 8, 9], &[7, 7]).is_err());
        assert!(check(&[7, 8, 9], &[8, 8, 8]).is_err());
    }

    #[test]
    fn repeated_lane_positions_preserve_the_replay_parameter() {
        let first = forja_sdk::Param::new(0..=16).unwrap();
        let other = forja_sdk::Param::new(0..=16).unwrap();
        let start = Dim::from(first.at(7));
        let next = start.clone().checked_add(1).unwrap();
        assert!(validate_rows(&[2], &[start.clone(), next], &[0, 0], 32).is_ok());
        assert!(validate_rows(&[2], &[start, other.at(8).into()], &[0, 0], 32).is_err());
    }
}
