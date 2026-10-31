use super::{Error, load_configured};
use crate::qwen38::serving::prefix::assert_boundaries;

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_canonical_short_boundaries_preserve_logits_and_state_bits()
-> Result<(), Box<dyn Error>> {
    let (runner, _) = load_configured(true, None, Some(8192)).await?;
    assert_boundaries(runner, &[63, 64, 65]).await
}

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_canonical_sparse_boundaries_preserve_logits_and_state_bits()
-> Result<(), Box<dyn Error>> {
    let (runner, _) = load_configured(true, None, Some(8192)).await?;
    assert_boundaries(runner, &[4096, 4097, 4161]).await
}

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_shared_canonical_parent_survives_two_live_branch_writers()
-> Result<(), Box<dyn Error>> {
    let (runner, _) = load_configured(true, None, Some(512)).await?;
    crate::qwen38::serving::prefix::assert_shared_parent(runner, 65).await
}

#[tokio::test]
#[ignore = "requires FORJA_MODELS and runs in the pre-push hook"]
async fn metal_real_recomputed_reply_preserves_cold_logits_and_every_state_plane()
-> Result<(), Box<dyn Error>> {
    let (runner, _) = load_configured(true, None, Some(512)).await?;
    crate::qwen38::serving::prefix::assert_recomputed_reply(runner).await
}
