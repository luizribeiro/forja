use crate::Model;
use forja_sdk::{Dim, Error, PagedVerifyTensors, Result};

fn inputs(model: &Model) -> Result<PagedVerifyTensors> {
    super::inputs(model)?
        .1
        .verify
        .ok_or_else(|| Error::loading("verify inputs are missing"))
}

pub(super) fn record(model: &mut Model, rows: u32, start: &Dim, lane: u32) -> Result<()> {
    if !(2..=crate::Qwen38::VERIFY_ROWS).contains(&rows) {
        return Err(Error::loading("verify row count is unsupported"));
    }
    let (_, shared) = super::inputs(model)?;
    let inputs = inputs(model)?;
    let words = shared.allow.shape()[1];
    let starts = (0..rows)
        .map(|row| start.clone().checked_add(row))
        .collect::<Result<Vec<_>>>()?;
    let tokens = inputs
        .tokens
        .narrow(0, start.clone().replay_constant(0), rows)?;
    let lanes = vec![lane; starts.len()];
    model
        .forward_decode_rows(&tokens, &starts, &lanes)?
        .copy_into(&mut inputs.logits.narrow(0, 0, rows)?)?;
    for (row, position) in (0..rows).zip(starts) {
        let anchor = position.clone().replay_constant(row);
        let logits = inputs.logits.narrow(0, anchor.clone(), 1)?;
        logits
            .sample_masked(
                &shared.sampling.narrow(0, lane, 1)?.reshape(&[5])?,
                &shared.allow.narrow(0, lane, 1)?.reshape(&[words])?,
                position,
            )?
            .copy_into(&mut inputs.selected.narrow(0, anchor, 1)?)?;
    }
    Ok(())
}

pub(super) fn output(model: &Model, rows: u32) -> Result<forja_sdk::BatchOutput> {
    let inputs = inputs(model)?;
    Ok(forja_sdk::BatchOutput {
        logits: inputs.logits.narrow(0, 0, rows)?,
        tokens: inputs.selected.narrow(0, 0, rows)?,
    })
}
