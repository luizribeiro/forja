use super::{
    Arc, Backend, LanePosition, StateCharge, StateSequenceHandle, StateSnapshot, compute,
    guest_error, invalid_handle,
};
use forja_core::HybridPrefixNamespace;

impl<B: Backend> StateSequenceHandle<B> {
    /// Attaches immutable carry to an unused sequence without checkpoint readers.
    ///
    /// # Errors
    /// Refuses foreign identities, used/pinned/busy sequences, feedback bindings or insufficient capacity.
    pub fn attach_snapshot(
        &self,
        image: Arc<StateSnapshot<B>>,
        namespace: HybridPrefixNamespace,
        reserved_tokens: u64,
    ) -> Result<(), compute::Error> {
        self.storage.check_usable()?;
        if image.namespace() != namespace
            || image.layout != *self.storage.pool.layout()
            || !Arc::ptr_eq(&image.backend, &self.storage.backend)
            || !Arc::ptr_eq(&image.charge.live, &self.storage.charge.live)
            || reserved_tokens <= image.position.tokens
        {
            return Err(invalid_handle("incompatible state snapshot attachment"));
        }
        let mut journal = self.lock();
        let mut attached = self.snapshot.lock().map_err(invalid_handle)?;
        if attached.is_some() {
            return Err(invalid_handle("state snapshot already attached"));
        }
        journal
            .attach_snapshot(
                Arc::clone(&image.prompt),
                LanePosition {
                    reserved_tokens,
                    ..image.position
                },
            )
            .map_err(invalid_handle)?;
        *attached = Some(image);
        Ok(())
    }

    /// Fingerprints every completed plane's raw payload, including non-cacheable state.
    /// The returned position belongs to the pinned version used for the digest.
    ///
    /// # Errors
    /// Refuses in-flight/uncertain state, unavailable scratch quota or backend failures.
    pub fn fingerprint(&self) -> Result<(LanePosition, [u8; 32]), compute::Error> {
        self.storage.check_usable()?;
        let journal = self.lock();
        journal.ensure_snapshot_idle().map_err(invalid_handle)?;
        let checkpoint = journal.checkpoint().map_err(invalid_handle)?;
        let record = journal
            .allocation_table()
            .map_err(invalid_handle)?
            .resolve(checkpoint.version())
            .map_err(invalid_handle)?;
        let position = journal.position;
        let attached = self.snapshot.lock().map_err(invalid_handle)?.clone();
        drop(journal);
        let layout = self.storage.pool.layout();
        let metadata = u64::try_from(layout.planes().len())
            .ok()
            .and_then(|n| n.checked_mul(32))
            .and_then(|n| n.checked_add(256))
            .and_then(|n| n.checked_add(layout.version_bytes()))
            .ok_or_else(|| crate::quota("state fingerprint extent overflow"))?;
        let _temporary = StateCharge::reserve(
            self.storage.backend.as_ref(),
            Arc::clone(&self.storage.charge.live),
            self.storage.limit,
            metadata,
        )?;
        let mut planes = Vec::new();
        planes
            .try_reserve_exact(layout.planes().len())
            .map_err(|_| crate::quota("state fingerprint allocation failed"))?;
        for plane in 0..layout.planes().len() {
            let source = match &attached {
                Some(image) => image.planes[plane].clone(),
                None => self.storage.version_view(plane, record)?,
            };
            planes.push(
                self.storage
                    .backend
                    .read_replay_output(&source)
                    .map_err(guest_error)?,
            );
        }
        let digest = forja_core::state_payload_digest(planes.iter().map(Vec::as_slice))
            .map_err(guest_error)?;
        Ok((position, digest))
    }

    pub(crate) fn materialize_snapshot(&self) -> Result<(), compute::Error> {
        let journal = self.lock();
        let mut attached = self.snapshot.lock().map_err(invalid_handle)?;
        let Some(image) = attached.as_ref() else {
            return Ok(());
        };
        let checkpoint = journal.checkpoint().map_err(invalid_handle)?;
        let record = journal
            .allocation_table()
            .map_err(invalid_handle)?
            .resolve(checkpoint.version())
            .map_err(invalid_handle)?;
        drop(journal);
        let _temporary = StateCharge::reserve(
            self.storage.backend.as_ref(),
            Arc::clone(&self.storage.charge.live),
            self.storage.limit,
            image.layout.version_bytes(),
        )?;
        for (plane, source) in image.planes.iter().enumerate() {
            let data = image
                .backend
                .read_replay_output(source)
                .map_err(guest_error)?;
            self.storage
                .backend
                .write(&self.storage.version_view(plane, record)?, &data)
                .map_err(guest_error)?;
        }
        *attached = None;
        Ok(())
    }
}
