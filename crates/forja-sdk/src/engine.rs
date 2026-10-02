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
    /// Layer indices available as router-logit taps.
    pub router_layers: Vec<u32>,
}

/// Optional model dimensions selected when an engine loads.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EngineLoadConfig {
    /// Number of leading decoder layers to load, or all layers when absent.
    pub num_hidden_layers: Option<u32>,
    /// Whether to capture and replay reusable graphs.
    pub replay: bool,
    /// Effective registered tuning names.
    pub tunings: Vec<String>,
    /// Effective fixed variant picks.
    pub fixed_variant_picks: Vec<FixedVariantPick>,
    /// Effective parameter-dependent variant picks.
    pub variant_rule_picks: Vec<VariantRulePick>,
}

impl Default for EngineLoadConfig {
    fn default() -> Self {
        Self {
            num_hidden_layers: None,
            replay: true,
            tunings: Vec::new(),
            fixed_variant_picks: Vec::new(),
            variant_rule_picks: Vec::new(),
        }
    }
}

/// One fixed engine dispatch-site selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FixedVariantPick {
    /// Stable engine-owned dispatch site.
    pub site: String,
    /// Stable backend algorithm name.
    pub name: String,
}

/// One parameter-dependent engine dispatch-site selection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariantRulePick {
    /// Stable engine-owned dispatch site.
    pub site: String,
    /// Replay parameter name.
    pub parameter: String,
    /// Complete ordered rule arms.
    pub arms: Vec<VariantPickArm>,
}

/// One inclusive arm in an engine variant rule.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariantPickArm {
    /// Inclusive lower bound.
    pub lo: u32,
    /// Inclusive upper bound.
    pub hi: u32,
    /// Stable backend algorithm name.
    pub name: String,
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
    /// Router scores with shape `[sequence, experts]` for declared router layers.
    pub router_logits: Vec<Tensor<f32>>,
}

/// Sampling parameters applied during decode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingParams {
    /// Logit temperature, where zero selects greedily.
    pub temperature: f32,
    /// Number of greatest logits retained, where zero disables top-k.
    pub top_k: u32,
    /// Cumulative probability retained after top-k.
    pub top_p: f32,
    /// Counter-based random seed.
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

/// Input to decode with an engine-retained feedback token.
pub struct DecodeInput {
    /// Tokens to process, or `None` to reuse the preceding selected token.
    pub tokens: Option<Tensor<u32>>,
    /// Position assigned to the first token, including a reused feedback token.
    pub start_pos: u32,
    /// Token sampling parameters.
    pub sampling: SamplingParams,
}

/// Device-resident outputs from greedy decode.
pub struct DecodeOutput {
    /// Last-position logits with shape `[vocab]`.
    pub logits: Tensor<f32>,
    /// Selected token id with shape `[1]`.
    pub token: Tensor<u32>,
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
    fn load(weights: &Weights<'_>, config: EngineLoadConfig) -> Result<Self>;

    /// Runs one unbatched model invocation.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is invalid or execution fails.
    fn step(&mut self, input: StepInput) -> Result<StepOutput>;

    /// Runs decode, optionally reusing the engine-retained token selected previously.
    ///
    /// # Errors
    ///
    /// Returns an error when this engine does not provide retained-token decode.
    fn decode(&mut self, _input: DecodeInput) -> Result<DecodeOutput> {
        Err(crate::Error::loading(
            "engine does not support retained-token decode",
        ))
    }
}
