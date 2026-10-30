use super::StateCell;
use crate::{Host, compute, core_dtype, invalid_handle};
use forja_core::{Backend, StateLayout, StatePlane};
use std::{any::Any, future::Future};
use wasmtime::component::Resource;

/// Opaque component resource; the guest can never obtain its plane tensors.
pub struct StateEntry {
    _cell: Box<dyn Any + Send + Sync>,
}

impl<B: Backend + Send + Sync + 'static> compute::HostStatePlane for Host<B> {
    fn create(
        &mut self,
        dtype: compute::Dtype,
        shape: Vec<u32>,
    ) -> impl Future<Output = wasmtime::Result<Result<Resource<StateEntry>, compute::Error>>> + Send
    {
        let result = (|| {
            self.check_handle_quota()?;
            let plane = StatePlane::new(core_dtype(dtype), shape).map_err(invalid_handle)?;
            let layout = StateLayout::new(vec![plane]).map_err(invalid_handle)?;
            let cell = StateCell::new(self.create_state_storage(layout, 1)?)?;
            let resource = self
                .table
                .push(StateEntry {
                    _cell: Box::new(cell),
                })
                .map_err(invalid_handle)?;
            self.live_handles += 1;
            Ok(resource)
        })();
        std::future::ready(Ok(result))
    }

    fn drop(
        &mut self,
        resource: Resource<StateEntry>,
    ) -> impl Future<Output = wasmtime::Result<()>> + Send {
        let result = self
            .table
            .delete(resource)
            .map(|_| {
                self.live_handles -= 1;
            })
            .map_err(wasmtime::Error::msg);
        std::future::ready(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Limits;
    use compute::HostStatePlane;
    #[tokio::test]
    async fn state_plane_handle_limit_refuses_new_resources_until_drop() {
        let mut host = Host::new(
            forja_cpu::CpuBackend::new(),
            Limits::new(65536, 8, 4096, 1, 4096),
        );
        let first = host
            .create(compute::Dtype::F32, vec![1, 1])
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(
            host.create(compute::Dtype::F32, vec![1, 1]).await.unwrap(),
            Err(compute::Error::Quota(_))
        ));
        HostStatePlane::drop(&mut host, first).await.unwrap();
        let next = host
            .create(compute::Dtype::F32, vec![1, 1])
            .await
            .unwrap()
            .unwrap();
        HostStatePlane::drop(&mut host, next).await.unwrap();
        assert_eq!(host.live_handles, 0);
    }
}
