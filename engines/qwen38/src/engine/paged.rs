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
    let tokens = if phase == BatchPhase::Prompt {
        inputs.prompt.narrow(0, 0, rows)?
    } else {
        set.slot().token(lane)?
    };
    let output = model.forward_paged(&tokens, start, lane, phase, false)?;
    let mut logits = inputs.logits.narrow(0, lane, 1)?;
    output
        .logits
        .reshape(logits.shape())?
        .copy_into(&mut logits)?;
    Ok(logits)
}

pub(super) fn select(model: &Model, lane: u32, position: &Dim) -> Result<Tensor<u32>> {
    let (set, inputs) = inputs(model)?;
    let words = inputs.allow.shape()[1];
    let selected = inputs.logits.narrow(0, lane, 1)?.sample_masked(
        &inputs.sampling.narrow(0, lane, 1)?.reshape(&[5])?,
        &inputs.allow.narrow(0, lane, 1)?.reshape(&[words])?,
        position.clone(),
    )?;
    let mut token = inputs.selected.narrow(0, lane, 1)?;
    selected.copy_into(&mut token)?;
    set.slot().store_feedback(lane, &selected)?;
    Ok(token)
}

pub(super) fn row_count(tokens: Option<&Tensor<u32>>) -> Result<u32> {
    match tokens.map(Tensor::shape) {
        None => Ok(1),
        Some([rows]) if *rows > 0 => Ok(*rows),
        _ => Err(Error::loading("decode requires a nonempty token vector")),
    }
}
