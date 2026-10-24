use std::error::Error;

pub(crate) struct Commands {
    pub(crate) reload: tokio::sync::mpsc::Receiver<()>,
    pub(crate) terminate: tokio::sync::mpsc::Receiver<()>,
}

#[cfg(unix)]
pub(crate) fn commands() -> Result<Commands, Box<dyn Error>> {
    use tokio::signal::unix::{SignalKind, signal};

    let mut reload = signal(SignalKind::hangup())?;
    let mut interrupt = signal(SignalKind::interrupt())?;
    let mut terminate = signal(SignalKind::terminate())?;
    let (reload_tx, reload_rx) = tokio::sync::mpsc::channel(1);
    let (terminate_tx, terminate_rx) = tokio::sync::mpsc::channel(1);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                value = terminate.recv() => {
                    if value.is_some() {
                        let _ = terminate_tx.send(()).await;
                    }
                    return;
                }
                value = interrupt.recv() => {
                    if value.is_some() {
                        let _ = terminate_tx.send(()).await;
                    }
                    return;
                }
                value = reload.recv() => {
                    if value.is_none() {
                        return;
                    }
                    let _ = reload_tx.try_send(());
                }
            }
        }
    });
    Ok(Commands {
        reload: reload_rx,
        terminate: terminate_rx,
    })
}

#[cfg(not(unix))]
pub(crate) fn commands() -> Result<Commands, Box<dyn Error>> {
    let (_reload_tx, reload_rx) = tokio::sync::mpsc::channel(1);
    let (terminate_tx, terminate_rx) = tokio::sync::mpsc::channel(1);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        let _ = terminate_tx.send(()).await;
    });
    Ok(Commands {
        reload: reload_rx,
        terminate: terminate_rx,
    })
}
