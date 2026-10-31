use crate::{Model, Qwen38};
use forja_sdk::{
    BatchPhase, BatchSelection, BatchShape, Dim, EngineLoadConfig, Error, Graph, Param, Result,
};

pub(crate) struct Catalog {
    context: u32,
    graphs: Vec<(BatchShape, Graph<()>)>,
}
impl Catalog {
    pub(crate) fn new(model: &mut Model, load: &EngineLoadConfig) -> Result<Self> {
        let context = load
            .max_context
            .ok_or_else(|| Error::loading("serving context is missing"))?;
        let mut catalog = Self {
            context,
            graphs: Vec::new(),
        };
        if load.replay {
            forja_sdk::eval()?;
            for shape in Qwen38::serving_shapes().into_iter().filter(|shape| {
                shape.phase == BatchPhase::Decode
                    && shape.selection == BatchSelection::Combined
                    && shape.bucket <= 4
            }) {
                catalog
                    .graphs
                    .push((shape, Self::capture(model, shape, context)?));
            }
        }
        Ok(catalog)
    }
    fn capture(model: &mut Model, shape: BatchShape, context: u32) -> Result<Graph<()>> {
        let end = context
            .checked_sub(1)
            .ok_or_else(|| Error::loading("serving context is empty"))?;
        let params = (0..shape.bucket)
            .map(|_| Param::new(0..=end))
            .collect::<Result<Vec<_>>>()?;
        let refs = params.iter().collect::<Vec<_>>();
        let set = model.serving_set()?;
        set.slot().capture(&refs, BatchPhase::Decode, || {
            for (lane, parameter) in params.iter().enumerate() {
                let lane =
                    u32::try_from(lane).map_err(|_| Error::loading("captured lane exceeds u32"))?;
                let start = Dim::from(parameter.at(0));
                drop(super::forward(model, 1, &start, lane, BatchPhase::Decode)?);
                drop(super::select(model, lane, &start, true)?);
            }
            Ok(())
        })
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
