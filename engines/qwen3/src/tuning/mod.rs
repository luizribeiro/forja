#[path = "apple-m3-ultra/final_norm.rs"]
pub(crate) mod final_norm;
#[path = "apple-m3-ultra/qk_norm_rope.rs"]
pub(crate) mod qk_norm_rope;
#[path = "apple-m3-ultra/residual_norm.rs"]
pub(crate) mod residual_norm;
#[path = "apple-m3-ultra/silu_mul.rs"]
pub(crate) mod silu_mul;

use forja_sdk::Tuning;

pub(crate) const REGISTRY: &[Tuning] = &[
    residual_norm::META,
    qk_norm_rope::META,
    silu_mul::META,
    final_norm::META,
];
