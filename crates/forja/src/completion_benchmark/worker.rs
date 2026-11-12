use std::{
    error::Error,
    io::{BufRead as _, Write as _},
    time::Instant,
};

use forja_config::{DevConfig, Profile};
use forja_serve::{ChatService, CompletionSession};
use serde_json::{Value, json};

use super::{Options, workload::Suite};

#[allow(clippy::too_many_arguments)]
pub(super) async fn run<S: ChatService>(
    options: &Options,
    config: &DevConfig,
    profile: &Profile,
    suite: &Suite,
    inputs: &Value,
    session: &CompletionSession<S>,
    device: &str,
    started: Instant,
) -> Result<(), Box<dyn Error>> {
    session.checkpoint().await?;
    let identity = json!({"protocol":1,"pid":std::process::id(),"runner_loads":1,
        "instance_loads":{"completion":1},"startup_seconds":started.elapsed().as_secs_f64()});
    println!("FORJA_BENCH_READY {identity}");
    std::io::stdout().flush()?;
    let mut expected = 1_u64;
    for line in std::io::stdin().lock().lines() {
        let request = crate::benchmark_worker::Request::parse(&line?, expected)?;
        let mut sample = options.clone();
        sample.json = request.output;
        let started = Instant::now();
        let mut document =
            super::measure(&sample, config, profile, suite, inputs, session, device).await?;
        document["worker"] = identity.clone();
        document["measurement_mode"]["execution"] = "resident-worker".into();
        super::write_record(
            &sample,
            document,
            config.backend.kind == forja_config::BackendKind::Metal,
        )?;
        let mut result = identity.clone();
        result["id"] = json!(request.id);
        result["wall_seconds"] = json!(started.elapsed().as_secs_f64());
        println!("FORJA_BENCH_RESULT {result}");
        std::io::stdout().flush()?;
        expected = expected
            .checked_add(1)
            .ok_or("worker request count overflow")?;
    }
    Ok(())
}
