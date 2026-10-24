use std::error::Error;

#[cfg(unix)]
pub(crate) async fn shutdown() -> Result<(), Box<dyn Error>> {
    use tokio::signal::unix::{SignalKind, signal};

    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
    let interrupt_sender = sender.clone();
    tokio::spawn(async move {
        if interrupt.recv().await.is_some() {
            let _ = interrupt_sender.send(()).await;
        }
    });
    tokio::spawn(async move {
        if terminate.recv().await.is_some() {
            let _ = sender.send(()).await;
        }
    });
    receiver
        .recv()
        .await
        .ok_or_else(|| "shutdown signal listeners closed".into())
}

#[cfg(not(unix))]
pub(crate) async fn shutdown() -> Result<(), Box<dyn Error>> {
    tokio::signal::ctrl_c().await?;
    Ok(())
}
