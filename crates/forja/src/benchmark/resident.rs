use super::{Bench, BenchPoint, EngineInfo, EngineRunner, Error, Path, load_benchmark_runner};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, serde::Serialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Phase {
    Single,
    Batch,
}

type Loaded = (EngineRunner<forja_metal::MetalBackend>, EngineInfo, String);

pub(super) struct Runners {
    retain: bool,
    loaded: std::collections::BTreeMap<Phase, Loaded>,
    pub(super) instance_loads: std::collections::BTreeMap<Phase, u64>,
    pub(super) loads: u64,
}

impl Runners {
    pub(super) const fn new(retain: bool) -> Self {
        Self {
            retain,
            loaded: std::collections::BTreeMap::new(),
            instance_loads: std::collections::BTreeMap::new(),
            loads: 0,
        }
    }

    pub(super) const fn retains(&self) -> bool {
        self.retain
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
