use crate::common::verify_runner as runner;
#[tokio::test]
async fn verify_declarations_refuse_unpaged_loading_before_guest_execution() -> wasmtime::Result<()>
{
    let mut runner = runner().await?;
    assert!(!runner.try_reserve_draft_lane(0)?);
    assert!(runner.try_release_draft_lane(0)?);
    let error = runner.load().await?.unwrap_err();
    assert!(
        matches!(error, forja_host::bindings::l9o::gpu::compute::Error::Layout(message) if message == "verify shapes require paged serving storage")
    );
    Ok(())
}
