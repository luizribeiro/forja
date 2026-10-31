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
