mod publication;
pub(crate) use publication::VerifyPublication;

use crate::{EngineRunner, Host, compute};
use forja_core::Backend;

impl<B: Backend + Send + Sync + 'static> EngineRunner<B> {
    /// Claims the shared state records before entering drained draft decoding.
    ///
    /// A quota or busy-record refusal returns `false` for ordinary decode fallback.
    /// Engines without protected state need no additional records.
    ///
    /// # Errors
    /// Returns missing-lane, storage or initialization failures.
    pub fn try_reserve_draft_lane(&self, lane: u32) -> Result<bool, compute::Error> {
        let Some(pool) = self.page_pool else {
            return Ok(false);
        };
        self.store.data().reserve_draft_lane(pool, lane)
    }

    /// Releases shared state records after a drafting lane's exit has drained.
    ///
    /// `false` means a live version or retained predecessor still occupies a spare;
    /// continue drained plain decoding before retrying.
    ///
    /// # Errors
    /// Returns missing-lane or storage failures.
    pub fn try_release_draft_lane(&self, lane: u32) -> Result<bool, compute::Error> {
        let Some(pool) = self.page_pool else {
            return Ok(true);
        };
        self.store.data().release_draft_lane(pool, lane)
    }
}

impl<B: Backend + 'static> Host<B> {
    fn reserve_draft_lane(&self, pool: u32, lane: u32) -> Result<bool, compute::Error> {
        self.paged_lane_prefix(pool, lane)?;
        self.lane_state(pool, lane)?
            .map_or(Ok(true), |sequence| sequence.try_reserve_spares())
    }

    fn release_draft_lane(&self, pool: u32, lane: u32) -> Result<bool, compute::Error> {
        self.paged_lane_prefix(pool, lane)?;
        self.lane_state(pool, lane)?
            .map_or(Ok(true), |sequence| sequence.try_release_spares())
    }
}

impl crate::EngineInfo {
    pub(crate) fn validate_verify_load(
        &self,
        config: &crate::EngineLoadConfig,
    ) -> Result<(), compute::Error> {
        if !config.paged_cache
            && self
                .serving
                .as_ref()
                .is_some_and(|serving| serving.verify_rows() != 0)
        {
            return Err(compute::Error::Layout(
                "verify shapes require paged serving storage".to_owned(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::verify_input_fixture;
    use forja_cpu::CpuBackend;

    #[test]
    fn runner_spare_routing_preserves_plain_decode_on_busy_or_quota_refusal() {
        let (mut host, pool, _) = verify_input_fixture(CpuBackend::new(), 3, 8);
        for lane in [0, 7] {
            host.bind_paged_lane(pool.rep(), lane, None, 33).unwrap();
        }
        assert!(host.reserve_draft_lane(pool.rep(), 7).unwrap());
        assert!(!host.reserve_draft_lane(pool.rep(), 0).unwrap());
        let peer = host.lane_state(pool.rep(), 0).unwrap().unwrap();
        let plain = peer.begin(1).unwrap();
        assert!(!host.reserve_draft_lane(pool.rep(), 0).unwrap());
        peer.cancel(&plain).unwrap();
        assert!(host.release_draft_lane(pool.rep(), 7).unwrap());
        assert!(host.reserve_draft_lane(pool.rep(), 0).unwrap());
        assert!(host.release_draft_lane(pool.rep(), 0).unwrap());
        assert!(host.reserve_draft_lane(pool.rep(), 8).is_err());
    }
}
