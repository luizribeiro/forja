use super::{Bench, BenchPoint, EngineInfo, EngineRunner, Error, Path, load_benchmark_runner};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Phase {
    Single,
    Batch,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(super) enum Prime {
    Prefill(usize),
    Decode(usize),
    Batch(u32),
}

type Loaded = (EngineRunner<forja_metal::MetalBackend>, EngineInfo, String);

pub(super) struct Runners {
    retain: bool,
    pub(super) batch_prefixes: std::collections::BTreeMap<usize, super::PagedPrefix>,
    pub(super) prefix_builds: u64,
    primed: std::collections::BTreeSet<Prime>,
    loaded: std::collections::BTreeMap<Phase, Loaded>,
    pub(super) instance_loads: std::collections::BTreeMap<Phase, u64>,
    pub(super) loads: u64,
    pub(super) times: super::timing::Timings,
    pub(super) prefixes: std::collections::BTreeMap<(Phase, usize), super::prefix::Prepared>,
}

impl Runners {
    pub(super) const fn new(retain: bool) -> Self {
        Self {
            retain,
            batch_prefixes: std::collections::BTreeMap::new(),
            prefix_builds: 0,
            primed: std::collections::BTreeSet::new(),
            loaded: std::collections::BTreeMap::new(),
            instance_loads: std::collections::BTreeMap::new(),
            loads: 0,
            times: super::timing::Timings::new(),
            prefixes: std::collections::BTreeMap::new(),
        }
    }

    pub(super) const fn retains(&self) -> bool {
        self.retain
    }

    pub(super) fn prefix_entries(&self) -> usize {
        self.prefixes.len() + self.batch_prefixes.len()
    }

    pub(super) fn prefix_built(&mut self) -> Result<(), Box<dyn Error>> {
        self.prefix_builds = self
            .prefix_builds
            .checked_add(1)
            .ok_or("prefix build count overflow")?;
        Ok(())
    }

    pub(super) fn warmup_count(&mut self, key: Prime, configured: usize) -> usize {
        if !self.retain {
            return configured;
        }
        usize::from(configured > 0 && self.primed.insert(key))
    }

    pub(super) fn prime_steps(&self, steps: usize) -> usize {
        if self.retain { steps.min(2) } else { steps }
    }

    pub(super) async fn acquire(
        &mut self,
        options: &Bench,
        point: &BenchPoint,
        component: &Path,
    ) -> Result<Loaded, Box<dyn Error>> {
        self.acquire_phase(options, point, component, Phase::Single)
            .await
    }

    pub(super) async fn acquire_phase(
        &mut self,
        options: &Bench,
        point: &BenchPoint,
        component: &Path,
        phase: Phase,
    ) -> Result<Loaded, Box<dyn Error>> {
        if let Some(loaded) = self.loaded.remove(&phase) {
            return Ok(loaded);
        }
        self.loads = self
            .loads
            .checked_add(1)
            .ok_or("runner load count overflow")?;
        let count = self.instance_loads.entry(phase).or_default();
        *count = count.checked_add(1).ok_or("instance load count overflow")?;
        load_benchmark_runner(options, point, component).await
    }

    pub(super) async fn release(&mut self, loaded: Loaded) -> Result<(), Box<dyn Error>> {
        self.release_phase(Phase::Single, loaded).await
    }

    pub(super) async fn release_phase(
        &mut self,
        phase: Phase,
        mut loaded: Loaded,
    ) -> Result<(), Box<dyn Error>> {
        if self.retain {
            loaded.0.wait_for_queued_submissions().await?;
            loaded
                .0
                .prepare_paged_request(None, 0)
                .map_err(|error| format!("resident request release failed: {error:?}"))?;
            let lanes = loaded
                .1
                .serving
                .as_ref()
                .and_then(|serving| serving.batch_shapes.iter().map(|shape| shape.bucket).max())
                .unwrap_or(1);
            for lane in 1..lanes {
                loaded
                    .0
                    .prepare_paged_lane(lane, None, 0)
                    .map_err(|error| format!("resident lane release failed: {error:?}"))?;
            }
            if self.loaded.insert(phase, loaded).is_some() {
                return Err("resident phase was returned twice".into());
            }
        }
        Ok(())
    }
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: u64,
    output: std::path::PathBuf,
}

impl Request {
    fn parse(line: &str, expected_id: u64) -> Result<Self, Box<dyn Error>> {
        let request: Self = serde_json::from_str(line)?;
        if request.id != expected_id || !request.output.is_absolute() || request.output.exists() {
            return Err("worker requires ordered requests and new absolute output paths".into());
        }
        Ok(request)
    }
}

fn validate_options(options: &Bench) -> Result<(), Box<dyn Error>> {
    if options.engines.len() != 1
        || options.points.len() != 1
        || options.points[0].selection.len() != 1
    {
        return Err("resident workers require one frozen engine, point and selection".into());
    }
    if options.points[0].selection != [forja_config::Selection::GpuPipelined] {
        return Err("prefix workers require gpu-pipelined selection".into());
    }
    if options.breakdown {
        return Err("prefix workers require unprofiled measurements".into());
    }
    Ok(())
}

pub(super) async fn run(options: &Bench) -> Result<(), Box<dyn Error>> {
    use std::io::{BufRead, Write};
    validate_options(options)?;
    let started = std::time::Instant::now();
    let mut runners = Runners::new(true);
    for phase in [Phase::Single, Phase::Batch] {
        if phase == Phase::Batch
            && options.mode.batch_measurements == crate::args::BatchMeasurements::Omit
        {
            continue;
        }
        let loaded = runners
            .acquire_phase(options, &options.points[0], &options.engines[0], phase)
            .await?;
        runners.release_phase(phase, loaded).await?;
    }
    println!(
        "FORJA_BENCH_READY {}",
        serde_json::json!({"protocol":2,"pid":std::process::id(),
        "runner_loads":runners.loads,"instance_loads":runners.instance_loads,"startup_seconds":started.elapsed().as_secs_f64()})
    );
    std::io::stdout().flush()?;
    let mut next = 1_u64;
    for line in std::io::stdin().lock().lines() {
        let request = Request::parse(&line?, next)?;
        let started = std::time::Instant::now();
        runners.times = super::timing::Timings::new();
        super::run_metal(options, &mut runners, Some(&request.output)).await?;
        println!(
            "FORJA_BENCH_RESULT {}",
            serde_json::json!({"id":request.id,"protocol":2,"prefix_entries":runners.prefix_entries(),"prefix_builds":runners.prefix_builds,
            "pid":std::process::id(),"runner_loads":runners.loads,
            "instance_loads":runners.instance_loads,"wall_seconds":started.elapsed().as_secs_f64()})
        );
        std::io::stdout().flush()?;
        next = next.checked_add(1).ok_or("worker request count overflow")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_workers_refuse_profiling_before_loading_a_runner() {
        let crate::args::Command::Bench(mut options) = crate::args::parse(
            ["bench", "--engine", "/engine.wasm", "--model", "/model"].map(str::to_owned),
        )
        .unwrap() else {
            panic!("expected benchmark options");
        };
        options.points[0].selection = vec![forja_config::Selection::GpuPipelined];
        validate_options(&options).unwrap();
        options.breakdown = true;
        assert!(
            validate_options(&options)
                .unwrap_err()
                .to_string()
                .contains("unprofiled")
        );
    }

    #[test]
    fn priming_is_bounded_per_shape_without_changing_classic_warmups() {
        let mut retained = Runners::new(true);
        assert_eq!(retained.warmup_count(Prime::Decode(33), 0), 0);
        for key in [
            Prime::Decode(33),
            Prime::Decode(65),
            Prime::Prefill(512),
            Prime::Batch(4),
        ] {
            assert_eq!(retained.warmup_count(key, 3), 1);
            assert_eq!(retained.warmup_count(key, 3), 0);
        }
        assert_eq!(retained.prime_steps(128), 2);
        let mut classic = Runners::new(false);
        assert_eq!(classic.warmup_count(Prime::Decode(33), 3), 3);
        assert_eq!(classic.warmup_count(Prime::Decode(33), 3), 3);
        assert_eq!(classic.prime_steps(128), 128);
    }

    #[test]
    fn worker_protocol_refuses_replays_config_changes_and_existing_outputs() {
        let output = std::env::temp_dir().join(format!("forja-worker-{}", std::process::id()));
        let line = serde_json::json!({"id":1,"output":output}).to_string();
        assert!(Request::parse(&line, 1).is_ok());
        assert!(Request::parse(&line, 2).is_err());
        assert!(Request::parse(r#"{"id":1,"output":"relative.json"}"#, 1).is_err());
        assert!(Request::parse(r#"{"id":1,"output":"/unused","reps":4}"#, 1).is_err());
        std::fs::write(&output, b"record").unwrap();
        assert!(Request::parse(&line, 1).is_err());
        std::fs::remove_file(output).unwrap();
    }
}
