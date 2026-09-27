use crate::{Result, Tensor, Weights};

/// Static metadata declared by an inference engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineInfo {
    /// Number of logits produced for the last input position.
    pub vocab: u32,
    /// Largest supported context length.
    pub max_context: u32,
    /// Layer indices available as hidden-state taps.
    pub tap_layers: Vec<u32>,
}

/// Input to one unbatched engine invocation.
pub struct StepInput {
    /// Token ids for this invocation.
    pub tokens: Tensor<u32>,
    /// Position assigned to the first token.
    pub start_pos: u32,
    /// Whether hidden-state taps should be returned.
    pub taps: bool,
}

/// Device-resident outputs from one engine invocation.
pub struct StepOutput {
    /// Last-position logits with shape `[vocab]`.
    pub logits: Tensor<f32>,
    /// Hidden states with shape `[sequence, hidden]` for declared tap layers.
    pub taps: Vec<Tensor<f32>>,
}

/// A model implementation exported through the component engine contract.
pub trait Engine: Sized + 'static {
    /// Returns static model metadata without loading weights.
    fn describe() -> EngineInfo;

    /// Constructs the model from a host-granted weight collection.
    ///
    /// # Errors
    ///
    /// Returns an error when required weights are missing or incompatible.
    fn load(weights: &Weights<'_>) -> Result<Self>;

    /// Runs one unbatched model invocation.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is invalid or execution fails.
    fn step(&mut self, input: StepInput) -> Result<StepOutput>;
}
