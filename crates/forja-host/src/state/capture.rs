use super::StateRecording;
use crate::{compute, invalid_handle};
use forja_core::{Backend, CompletionResource, ReplayBindings, StateAccess, Tensor};
use std::sync::Arc;

/// Protected graph bindings whose ownership excludes a request's sequence reservation.
#[derive(Clone)]
pub struct StateCapture {
    access: StateAccess,
    planes: Vec<[Tensor; 2]>,
    metadata: Vec<Tensor>,
    updated: Vec<bool>,
    _owner: CompletionResource,
}

impl std::fmt::Debug for StateCapture {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StateCapture")
            .field("access", &self.access)
            .finish_non_exhaustive()
    }
}

impl<B: Backend + Send + Sync + 'static> StateRecording<B> {
    /// Retains protected backing and metadata without retaining this sequence's allocation lease.
    ///
    /// # Errors
    /// Returns an error if any protected plane view cannot be constructed.
    pub fn capture(&self) -> Result<StateCapture, compute::Error> {
        let planes = (0..self.updated.len())
            .map(|plane| {
                Ok([
                    self.storage.version_view(plane, self.access.source())?,
                    self.storage
                        .version_view(plane, self.access.destination())?,
                ])
            })
            .collect::<Result<_, compute::Error>>()?;
        Ok(StateCapture {
            access: self.access,
            planes,
            metadata: self.metadata.to_vec(),
            updated: self.updated.clone(),
            _owner: CompletionResource::new(Arc::new((
                Arc::clone(&self.storage),
                Arc::clone(&self.metadata),
            ))),
        })
    }
}

impl StateCapture {
    /// Binds a fresh transaction to a captured graph's protected operands.
    ///
    /// # Errors
    /// Refuses incompatible planes or a duplicate update in a different transaction.
    pub fn bind<B: Backend>(
        &self,
        recording: &mut StateRecording<B>,
    ) -> Result<ReplayBindings, compute::Error> {
        let same_recording = self.metadata.as_slice() == &**recording.metadata;
        if self.planes.len() != recording.updated.len()
            || self.metadata.len() != recording.metadata.len()
            || (!same_recording
                && self
                    .updated
                    .iter()
                    .zip(&recording.updated)
                    .any(|(old, new)| *old && *new))
        {
            return Err(invalid_handle(
                "captured state does not match the transaction",
            ));
        }
        let mut bindings = ReplayBindings::default();
        bindings
            .state(self.access, recording.access)
            .map_err(invalid_handle)?;
        for (plane, captured) in self.planes.iter().enumerate() {
            for (old, record) in captured
                .iter()
                .zip([recording.access.source(), recording.access.destination()])
            {
                bindings
                    .view(old.clone(), recording.storage.version_view(plane, record)?)
                    .map_err(invalid_handle)?;
            }
        }
        for (old, new) in self.metadata.iter().zip(recording.metadata.iter()) {
            bindings
                .buffer(old.buffer(), new.buffer())
                .map_err(invalid_handle)?;
        }
        for (current, updated) in recording.updated.iter_mut().zip(&self.updated) {
            *current |= updated;
        }
        Ok(bindings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Host, Limits};
    use forja_core::{DType, GraphLimits, GraphTemplate, ParamSpace, StateLayout, StatePlane};

    fn rebind_versions<B: Backend + Send + Sync + 'static>(backend: B) {
        let host = Host::new(backend, Limits::new(65536, 8, 4096, 128, 4096));
        let layout =
            StateLayout::new(vec![StatePlane::new(DType::F32, vec![3, 7]).unwrap()]).unwrap();
        let storage = host.create_state_storage(layout, 1).unwrap();
        let sequence = storage.reserve().unwrap();
        let input = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        let weights = host.backend.alloc(DType::F32, &[4, 7]).unwrap();
        let output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
        host.backend
            .write(&input, &2_f32.to_le_bytes().repeat(7))
            .unwrap();
        host.backend
            .write(&weights, &1_f32.to_le_bytes().repeat(28))
            .unwrap();
        let mut recording = storage
            .record(&sequence, sequence.begin(1).unwrap())
            .unwrap();
        recording
            .causal_convolution(0, &input, &weights, &output)
            .unwrap();
        let capture = recording.capture().unwrap();
        let mut graph =
            GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), GraphLimits::default());
        for dispatch in recording.commands.dispatches() {
            graph.record_validated(dispatch.clone()).unwrap();
        }
        recording.submit().unwrap().finish().unwrap();
        let mut pending = Vec::new();
        let mut versions = std::collections::HashSet::from([capture.access.destination()]);
        for (serial, expected) in [(2, 4_f32), (3, 6.0), (4, 8.0)] {
            let mut recording = storage
                .record(&sequence, sequence.begin(serial).unwrap())
                .unwrap();
            versions.insert(recording.access.destination());
            let bank_output = host.backend.alloc(DType::F32, &[1, 7]).unwrap();
            let mut bindings = capture.bind(&mut recording).unwrap();
            bindings
                .buffer(output.buffer(), bank_output.buffer())
                .unwrap();
            recording.commands = bindings
                .apply(
                    graph
                        .instantiate(&graph.values(Vec::new()).unwrap())
                        .unwrap(),
                )
                .unwrap();
            pending.push((recording.submit().unwrap(), bank_output, expected));
            if serial >= 3 {
                for (submission, output, expected) in pending.drain(..) {
                    submission.finish().unwrap();
                    assert_eq!(
                        host.backend.read(&output).unwrap(),
                        expected.to_le_bytes().repeat(7)
                    );
                }
            }
        }
        assert_eq!(versions.len(), 3);
        drop(sequence);
        let replacement = storage.reserve().unwrap();
        drop(replacement);
        drop(capture);
    }

    #[test]
    fn captured_state_rebinds_all_three_versions_without_retaining_the_lane() {
        rebind_versions(forja_cpu::CpuBackend::new());
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_captured_state_rebinds_all_three_versions_without_retaining_the_lane() {
        rebind_versions(forja_metal::MetalBackend::new().unwrap());
    }
}
