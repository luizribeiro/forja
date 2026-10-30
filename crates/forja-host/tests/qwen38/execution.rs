use forja_core::Backend;
use forja_sdk::Tensor;
use std::error::Error;

pub(super) struct Output {
    pub(super) logits: Vec<f32>,
    pub(super) taps: Vec<Vec<f32>>,
}
pub(super) trait Execution {
    async fn forward(
        &mut self,
        tokens: &[u32],
        start: u32,
        taps: bool,
    ) -> Result<Output, Box<dyn Error>>;
}
impl Execution for qwen38::Model {
    fn forward(
        &mut self,
        tokens: &[u32],
        start: u32,
        taps: bool,
    ) -> impl std::future::Future<Output = Result<Output, Box<dyn Error>>> {
        std::future::ready(native_forward(self, tokens, start, taps))
    }
}
fn native_forward(
    model: &mut qwen38::Model,
    tokens: &[u32],
    start: u32,
    taps: bool,
) -> Result<Output, Box<dyn Error>> {
    let output = qwen38::Model::forward(
        model,
        &Tensor::from_slice(tokens, &[u32::try_from(tokens.len())?])?,
        start,
        taps,
    )?;
    Ok(Output {
        logits: output.logits.to_vec()?,
        taps: output
            .taps
            .iter()
            .map(Tensor::to_vec)
            .collect::<Result<_, _>>()?,
    })
}
impl<B: Backend + Send + Sync + 'static> Execution for forja_host::EngineRunner<B> {
    async fn forward(
        &mut self,
        tokens: &[u32],
        start: u32,
        taps: bool,
    ) -> Result<Output, Box<dyn Error>> {
        let output = self
            .step(forja_host::EngineStep {
                phase: forja_host::BatchPhase::Prompt,
                tokens: tokens.to_vec(),
                start_pos: start,
                taps,
            })
            .await??;
        let logits = golden_fixtures::decode_f32_le(&self.read(&output.logits).await?)?;
        let mut taps = Vec::new();
        for tap in &output.taps {
            taps.push(golden_fixtures::decode_f32_le(&self.read(tap).await?)?);
        }
        Ok(Output { logits, taps })
    }
}
