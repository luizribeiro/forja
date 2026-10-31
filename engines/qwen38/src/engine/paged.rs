mod catalog;
pub(super) use catalog::Catalog;

use crate::Model;
use forja_sdk::{BatchPhase, Dim, Error, PagedCacheSet, PagedInputTensors, Result, Tensor};
use std::rc::Rc;

fn inputs(model: &Model) -> Result<(Rc<PagedCacheSet>, PagedInputTensors)> {
    let set = model.serving_set()?;
    let inputs = set
        .slot()
        .tensors()?
        .inputs
        .ok_or_else(|| Error::loading("serving inputs are missing"))?;
    Ok((set, inputs))
}

pub(super) fn forward(
    model: &mut Model,
    rows: u32,
    start: &Dim,
    lane: u32,
    phase: BatchPhase,
) -> Result<Tensor<f32>> {
    let (set, inputs) = inputs(model)?;
    let anchor = start.clone().replay_constant(0);
    let tokens = if phase == BatchPhase::Prompt {
        inputs.prompt.narrow(0, anchor, rows)?
    } else {
        set.slot().token(lane)?.narrow(0, anchor, 1)?
    };
    let output = model.forward_paged(&tokens, start, lane, phase, false)?;
    let mut logits = inputs.logits.narrow(0, lane, 1)?;
    output
        .logits
        .reshape(logits.shape())?
        .copy_into(&mut logits)?;
    Ok(logits)
}

fn forward_decode_rows(model: &mut Model, starts: &[Dim]) -> Result<()> {
    let (set, inputs) = inputs(model)?;
    let batch =
        u32::try_from(starts.len()).map_err(|_| Error::loading("decode lane count exceeds u32"))?;
    let tokens = Tensor::<u32>::zeros(&[batch])?;
    for (lane, start) in starts.iter().enumerate() {
        let lane = u32::try_from(lane).map_err(|_| Error::loading("decode lane exceeds u32"))?;
        set.slot()
            .token(lane)?
            .narrow(0, start.clone().replay_constant(0), 1)?
            .copy_into(&mut tokens.narrow(0, lane, 1)?)?;
    }
    model
        .forward_decode_rows(&tokens, starts)?
        .copy_into(&mut inputs.logits.narrow(0, 0, batch)?)?;
    Ok(())
}

pub(super) fn select(
    model: &Model,
    lane: u32,
    position: &Dim,
    feedback: bool,
) -> Result<Tensor<u32>> {
    let (set, inputs) = inputs(model)?;
    let words = inputs.allow.shape()[1];
    let row = position.clone().replay_constant(lane);
    let selected = inputs.logits.narrow(0, row, 1)?.sample_masked(
        &inputs.sampling.narrow(0, lane, 1)?.reshape(&[5])?,
        &inputs.allow.narrow(0, lane, 1)?.reshape(&[words])?,
        position.clone(),
    )?;
    let mut token = inputs.selected.narrow(0, lane, 1)?;
    selected.copy_into(&mut token)?;
    if feedback {
        set.slot().store_feedback(lane, &selected)?;
    }
    Ok(token)
}

pub(super) fn row_count(tokens: Option<&Tensor<u32>>) -> Result<u32> {
    match tokens.map(Tensor::shape) {
        None => Ok(1),
        Some([rows]) if *rows > 0 => Ok(*rows),
        _ => Err(Error::loading("decode requires a nonempty token vector")),
    }
}

pub(super) fn run_batch(
    model: &mut Model,
    input: &forja_sdk::BatchInput<'_>,
    selection: bool,
) -> Result<forja_sdk::BatchOutput> {
    let prompt = input.shape.phase == BatchPhase::Prompt;
    if !prompt {
        let starts = input
            .lanes
            .iter()
            .map(|lane| Dim::from(lane.as_ref().map_or(0, |lane| lane.start_pos)))
            .collect::<Vec<_>>();
        forward_decode_rows(model, &starts)?;
    }
    for (index, active) in input.lanes.iter().enumerate() {
        let index = u32::try_from(index).map_err(|_| Error::loading("batch index exceeds u32"))?;
        let lane = if prompt {
            active
                .as_ref()
                .ok_or_else(|| Error::loading("prompt lane is missing"))?
                .slot
        } else {
            index
        };
        let start = active.as_ref().map_or(0, |lane| lane.start_pos);
        let rows = if prompt {
            row_count(active.as_ref().and_then(|lane| lane.tokens.as_ref()))?
        } else {
            1
        };
        let position = start
            .checked_add(rows)
            .and_then(|end| end.checked_sub(1))
            .ok_or_else(|| Error::loading("batch position overflowed"))?;
        if prompt {
            drop(forward(
                model,
                rows,
                &start.into(),
                lane,
                input.shape.phase,
            )?);
        }
        if selection {
            drop(select(model, lane, &position.into(), active.is_some())?);
        }
    }
    let base = if prompt {
        input
            .lanes
            .first()
            .and_then(Option::as_ref)
            .ok_or_else(|| Error::loading("prompt lane is missing"))?
            .slot
    } else {
        0
    };
    forja_sdk::eval()?;
    batch_output(model, base, input.shape.bucket)
}

pub(super) fn batch_output(
    model: &Model,
    base: u32,
    bucket: u32,
) -> Result<forja_sdk::BatchOutput> {
    let (_, inputs) = inputs(model)?;
    Ok(forja_sdk::BatchOutput {
        logits: inputs.logits.narrow(0, base, bucket)?,
        tokens: inputs.selected.narrow(0, base, bucket)?,
    })
}

pub(super) fn select_batch(
    model: &Model,
    input: &forja_sdk::BatchSelectInput<'_>,
) -> Result<forja_sdk::BatchOutput> {
    for (lane, active) in input.lanes.iter().enumerate() {
        let lane = u32::try_from(lane).map_err(|_| Error::loading("selection lane exceeds u32"))?;
        let position = active.as_ref().map_or(0, |active| active.start_pos);
        drop(select(model, lane, &position.into(), active.is_some())?);
    }
    forja_sdk::eval()?;
    batch_output(model, 0, input.shape.bucket)
}
