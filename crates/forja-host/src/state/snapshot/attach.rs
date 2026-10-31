use super::{
    Arc, Backend, LanePosition, StateCharge, StateSequenceHandle, StateSnapshot, compute,
    guest_error, invalid_handle,
};
use forja_core::HybridPrefixNamespace;

impl<B: Backend> StateSequenceHandle<B> {
    /// Attaches immutable carry to a pristine sequence without copying its planes.
    ///
    /// # Errors
    /// Refuses foreign hosts, layouts, namespaces, busy sequences or insufficient capacity.
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
