use forja_sdk::{EngineLoadConfig, EngineTunings, Result};

use crate::tuning;

#[derive(Clone, Copy, Default)]
pub(crate) enum Slot {
    #[default]
    Default,
    Tuned,
}

impl Slot {
    pub(crate) const fn is_tuned(self) -> bool {
        matches!(self, Self::Tuned)
    }

    fn selected(active: bool) -> Self {
        if active { Self::Tuned } else { Self::Default }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Slots {
    pub(crate) residual_norm: Slot,
    pub(crate) qk_norm_rope: Slot,
    pub(crate) silu_mul: Slot,
    pub(crate) final_norm: Slot,
}

impl Slots {
    pub(crate) fn new(config: &EngineLoadConfig) -> Result<Self> {
        let selected = EngineTunings::new(config, tuning::REGISTRY)?;
        Ok(Self {
            residual_norm: Slot::selected(selected.contains("residual-norm")),
            qk_norm_rope: Slot::selected(selected.contains("qk-norm-rope")),
            silu_mul: Slot::selected(selected.contains("silu-mul")),
            final_norm: Slot::selected(selected.contains("final-norm")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_use_verified_unfused_slots() {
        let slots = Slots::new(&EngineLoadConfig::default()).unwrap();
        assert!(!slots.residual_norm.is_tuned());
        assert!(!slots.qk_norm_rope.is_tuned());
        assert!(!slots.silu_mul.is_tuned());
        assert!(!slots.final_norm.is_tuned());
    }

    #[test]
    fn profile_set_selects_every_tuned_slot() {
        let config = EngineLoadConfig {
            tunings: ["residual-norm", "qk-norm-rope", "silu-mul", "final-norm"]
                .map(str::to_owned)
                .to_vec(),
            ..EngineLoadConfig::default()
        };
        let slots = Slots::new(&config).unwrap();
        assert!(slots.residual_norm.is_tuned());
        assert!(slots.qk_norm_rope.is_tuned());
        assert!(slots.silu_mul.is_tuned());
        assert!(slots.final_norm.is_tuned());
    }
}
