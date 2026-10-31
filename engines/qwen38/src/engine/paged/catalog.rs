use crate::{Model, Qwen38};
use forja_sdk::{
    BatchPhase, BatchSelection, BatchShape, Dim, EngineLoadConfig, Error, Graph, Param, Result,
};

pub(crate) struct Catalog {
    context: u32,
    enabled: bool,
    tail: Option<(u32, Graph<()>)>,
    graphs: Vec<(BatchShape, Graph<()>)>,
}
impl Catalog {
    pub(crate) fn new(model: &mut Model, load: &EngineLoadConfig) -> Result<Self> {
        let context = load
            .max_context
            .ok_or_else(|| Error::loading("serving context is missing"))?;
        let mut catalog = Self {
            context,
            enabled: load.replay,
            tail: None,
            graphs: Vec::new(),
        };
        if load.replay {
            forja_sdk::eval()?;
            for shape in Qwen38::serving_shapes()
                .into_iter()
                .filter(|shape| match shape.phase {
                    BatchPhase::Decode => shape.bucket <= 4,
                    BatchPhase::Prompt => shape.tokens <= context,
                })
            {
                catalog
                    .graphs
                    .push((shape, Self::capture(model, shape, context)?));
            }
        }
        Ok(catalog)
    }
    fn capture(model: &mut Model, shape: BatchShape, context: u32) -> Result<Graph<()>> {
        let end = context
            .checked_sub(shape.tokens)
            .ok_or_else(|| Error::loading("serving context is empty"))?;
        let params = (0..shape.bucket)
            .map(|_| Param::new(0..=end))
            .collect::<Result<Vec<_>>>()?;
        let refs = params.iter().collect::<Vec<_>>();
        let set = model.serving_set()?;
        let final_offset = shape
            .tokens
            .checked_sub(1)
            .ok_or_else(|| Error::loading("captured span is empty"))?;
        set.slot().capture(&refs, shape.phase, || {
            let starts = params
                .iter()
                .map(|parameter| Dim::from(parameter.at(0)))
                .collect::<Vec<_>>();
            if shape.phase == BatchPhase::Decode && shape.selection != BatchSelection::Masked {
                super::forward_decode_rows(model, &starts)?;
            }
            for (lane, start) in starts.iter().enumerate() {
                let lane =
                    u32::try_from(lane).map_err(|_| Error::loading("captured lane exceeds u32"))?;
                if shape.phase == BatchPhase::Prompt && shape.selection != BatchSelection::Masked {
                    drop(super::forward(
                        model,
                        shape.tokens,
                        start,
                        lane,
                        shape.phase,
                    )?);
                }
                if shape.selection != BatchSelection::Forward {
                    drop(super::select(
                        model,
                        lane,
                        &start.clone().checked_add(final_offset)?,
                        true,
                    )?);
                }
            }
            Ok(())
        })
    }
    pub(crate) fn batch(
        &mut self,
        model: &mut Model,
        input: &forja_sdk::BatchInput<'_>,
        selection: bool,
    ) -> Result<forja_sdk::BatchOutput> {
        let prompt = input.shape.phase == BatchPhase::Prompt;
        let mut shape = input.shape;
        let base = if prompt {
            let lane = input
                .lanes
                .first()
                .and_then(Option::as_ref)
                .ok_or_else(|| Error::loading("prompt lane is missing"))?;
            shape.tokens = super::row_count(lane.tokens.as_ref())?;
            lane.slot
        } else {
            0
        };
        if self.run(
            model,
            shape,
            &positions(&input.lanes, |lane| lane.start_pos),
        )? {
            super::batch_output(model, base, input.shape.bucket)
        } else {
            super::run_batch(model, input, selection)
        }
    }
    pub(crate) fn select_batch(
        &self,
        model: &Model,
        input: &forja_sdk::BatchSelectInput<'_>,
    ) -> Result<forja_sdk::BatchOutput> {
        if self.replay(input.shape, &positions(&input.lanes, |lane| lane.start_pos))? {
            super::batch_output(model, 0, input.shape.bucket)
        } else {
            super::select_batch(model, input)
        }
    }
    pub(crate) fn run(
        &mut self,
        model: &mut Model,
        mut shape: BatchShape,
        positions: &[u32],
    ) -> Result<bool> {
        if !self.enabled {
            return Ok(false);
        }
        if shape.phase == BatchPhase::Decode {
            return self.replay(shape, positions);
        }
        let [start] = positions else {
            return Err(Error::loading("prompt replay requires one position"));
        };
        if !(1..=2048).contains(&shape.tokens)
            || !start.is_multiple_of(crate::PREFILL_BLOCK_TOKENS)
            || start
                .checked_add(shape.tokens)
                .is_none_or(|end| end > self.context)
        {
            return Err(Error::loading("invalid canonical prompt replay span"));
        }
        shape.selection = BatchSelection::Combined;
        if self.replay(shape, positions)? {
            return Ok(true);
        }
        if self
            .tail
            .as_ref()
            .is_none_or(|(rows, _)| *rows != shape.tokens)
        {
            self.tail = None;
            self.tail = Some((shape.tokens, Self::capture(model, shape, self.context)?));
        }
        self.tail
            .as_ref()
            .ok_or_else(|| Error::loading("prompt tail capture is missing"))?
            .1
            .replay(positions)?;
        Ok(true)
    }
    pub(crate) fn replay(&self, shape: BatchShape, positions: &[u32]) -> Result<bool> {
        let Some((_, graph)) = self.graphs.iter().find(|(key, _)| *key == shape) else {
            return Ok(false);
        };
        if positions.iter().any(|position| *position >= self.context) {
            return Err(Error::loading("decode position exceeds serving context"));
        }
        graph.replay(positions)?;
        Ok(true)
    }
}

fn positions<T>(lanes: &[Option<T>], start: impl Fn(&T) -> u32) -> Vec<u32> {
    lanes
        .iter()
        .map(|lane| lane.as_ref().map_or(0, &start))
        .collect()
}
