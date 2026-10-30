use crate::Config;
use forja_sdk::{
    DType, Error, FloatElement, PagedCacheSet, Result, Tensor, state::StatePlane,
    target::metal::Variant,
};

pub(super) enum Carry {
    Eager {
        history: StatePlane,
        recurrent: StatePlane,
    },
    Slotted {
        cache: std::rc::Rc<PagedCacheSet>,
        planes: [u32; 2],
        lane: u32,
    },
}

impl Carry {
    pub(super) fn new(config: Config) -> Result<Self> {
        Ok(Self::Eager {
            history: StatePlane::new(DType::BF16, &[config.conv_width - 1, config.qkv()?])?,
            recurrent: StatePlane::new(
                DType::F32,
                &[config.value_heads, config.key_dim, config.value_dim],
            )?,
        })
    }

    pub(super) fn slotted(cache: std::rc::Rc<PagedCacheSet>, first: u32) -> Result<Self> {
        let next = first
            .checked_add(1)
            .ok_or_else(|| Error::loading("state plane overflow"))?;
        cache.state_at(0, first)?;
        cache.state_at(0, next)?;
        Ok(Self::Slotted {
            cache,
            planes: [first, next],
            lane: 0,
        })
    }

    pub(super) const fn is_slotted(&self) -> bool {
        matches!(self, Self::Slotted { .. })
    }

    pub(super) fn convolve<T: FloatElement>(
        &mut self,
        input: &Tensor<T>,
        weights: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        match self {
            Self::Eager { history, .. } => history.causal_convolution(input, weights),
            Self::Slotted {
                cache,
                planes,
                lane,
            } => cache
                .state_at(*lane, planes[0])?
                .causal_convolution(input, weights),
        }
    }

    pub(super) fn delta<T: FloatElement>(
        &mut self,
        qk: [&Tensor<f32>; 2],
        value: &Tensor<T>,
        decay: &Tensor<f32>,
        beta: &Tensor<T>,
        block: bool,
        pick: Option<&Variant>,
    ) -> Result<Tensor<T>> {
        match self {
            Self::Eager { recurrent, .. } if block => {
                recurrent.delta_rule_block(qk[0], qk[1], value, decay, beta, pick)
            }
            Self::Eager { recurrent, .. } => recurrent.delta_rule(qk[0], qk[1], value, decay, beta),
            Self::Slotted {
                cache,
                planes,
                lane,
            } => {
                let state = cache.state_at(*lane, planes[1])?;
                if block {
                    state.delta_rule_block(qk[0], qk[1], value, decay, beta, pick)
                } else {
                    state.delta_rule(qk[0], qk[1], value, decay, beta)
                }
            }
        }
    }
}
