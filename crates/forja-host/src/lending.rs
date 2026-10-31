use crate::{Backend, Host, TensorEntry, compute, invalid_handle, state::StateCharge};
use std::sync::Arc;
use wasmtime::component::Resource;

const REFERENCE_BYTES: u64 = 1024;

impl<B: Backend + 'static> Host<B> {
    pub(crate) fn reserve_reference(&self) -> Result<Arc<StateCharge>, compute::Error> {
        self.reserve_references(1)
    }

    pub(crate) fn reserve_references(
        &self,
        count: usize,
    ) -> Result<Arc<StateCharge>, compute::Error> {
        let bytes = u64::try_from(count.max(1))
            .ok()
            .and_then(|count| count.checked_mul(REFERENCE_BYTES))
            .ok_or_else(|| crate::quota("reference metadata size overflow"))?;
        StateCharge::reserve(
            self.backend.as_ref(),
            Arc::clone(&self.live_bytes),
            self.limits.live_bytes,
            bytes,
        )
        .map(Arc::new)
    }

    /// Lends a validated host view with separately charged metadata and optional reduced rights.
    ///
    /// # Errors
    /// Refuses invalid views, unavailable resources or exhausted byte/handle quota.
    pub fn lend_view(
        &mut self,
        resource: &Resource<TensorEntry>,
        operation: compute::ViewOp,
        read_only: bool,
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        let charge = self
            .entry(resource)?
            .reference_charge
            .is_none()
            .then(|| self.reserve_reference())
            .transpose()?;
        let alias = self.view(resource, operation)?;
        let result = self
            .table
            .get_mut(&alias)
            .map_err(invalid_handle)
            .map(|entry| {
                if let Some(charge) = charge {
                    entry.reference_charge = Some(charge);
                }
                if read_only {
                    entry.tensor = entry.tensor.read_only();
                }
            });
        if let Err(error) = result {
            let _ = self.drop_tensor(alias);
            return Err(error);
        }
        Ok(alias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use std::sync::atomic::Ordering;

    #[test]
    fn lent_views_charge_descendants_and_retirement_without_escalating_rights() {
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(8192, 8, 4096, 32, 4096),
        );
        let input = host.alloc(compute::Dtype::U32, &[7]).unwrap();
        host.write(
            &input,
            &[17_u32; 7]
                .into_iter()
                .flat_map(u32::to_ne_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let before = host.live_bytes.load(Ordering::Acquire);
        host.limits.live_bytes = before;
        assert!(matches!(
            host.lend_view(&input, compute::ViewOp::Reshape(vec![7]), true),
            Err(compute::Error::Quota(_))
        ));
        assert_eq!(host.live_handles, 1);
        assert_eq!(host.live_bytes.load(Ordering::Acquire), before);
        host.limits.live_bytes = 8192;
        let loan = host
            .lend_view(&input, compute::ViewOp::Reshape(vec![7]), true)
            .unwrap();
        assert!(host.write(&loan, &[0; 28]).is_err());
        let child = host.view(&loan, compute::ViewOp::Reshape(vec![7])).unwrap();
        assert_eq!(
            host.live_bytes.load(Ordering::Acquire),
            before + 2 * REFERENCE_BYTES
        );
        assert!(!host.entry(&child).unwrap().tensor.is_writable());
        let parameters = host
            .params(vec![compute::ParamRange { lo: 0, hi: 6 }])
            .unwrap();
        let symbolic = host
            .view_param(
                &loan,
                &parameters,
                vec![compute::ParamSlice {
                    start: compute::Affine {
                        param: Some(0),
                        scale: 1,
                        offset: 0,
                    },
                    len: compute::Affine {
                        param: None,
                        scale: 0,
                        offset: 1,
                    },
                    step: 1,
                }],
            )
            .unwrap();
        assert_eq!(
            host.live_bytes.load(Ordering::Acquire),
            before + 3 * REFERENCE_BYTES
        );
        assert!(!host.entry(&symbolic).unwrap().tensor.is_writable());
        host.drop_tensor(symbolic).unwrap();

        let retained = host.entry(&child).unwrap().clone();
        host.drop_tensor(loan).unwrap();
        host.drop_tensor(child).unwrap();
        host.drop_tensor(input).unwrap();
        assert_eq!(
            host.live_bytes.load(Ordering::Acquire),
            before + REFERENCE_BYTES
        );
        host.release_retained(vec![retained]).unwrap();
        assert_eq!(host.live_bytes.load(Ordering::Acquire), 0);
    }
}
