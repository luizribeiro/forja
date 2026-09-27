//! Trusted host integration for running Forja guest components.

#![allow(clippy::manual_async_fn)]

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use forja_core::{Backend, BackendError, DType, LayoutError, Slice, Tensor, ViewOp};
use wasmtime::component::{Resource, ResourceTable};
use wasmtime::{Engine, Store, StoreLimits, StoreLimitsBuilder};

/// Host bindings for the guest-facing compute interface.
#[allow(missing_docs)]
pub mod bindings {
    wasmtime::component::bindgen!({
        path: "../../wit",
        world: "host",
        imports: { default: async | trappable },
        require_store_data_send: true,
        with: {
            "l9o:gpu/compute.tensor": crate::TensorEntry,
        },
    });
}

use bindings::l9o::gpu::compute;

/// Resource limits applied before backend work or component allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Limits {
    live_bytes: u64,
    tensor_rank: usize,
    tensor_elements: u64,
    live_tensor_handles: usize,
    read_bytes: u64,
    guest_memory_bytes: usize,
    table_elements: usize,
    instances: usize,
}

impl Limits {
    /// Creates tensor limits with a 4 GiB guest-memory default.
    #[must_use]
    pub const fn new(
        max_live_bytes: u64,
        max_tensor_rank: usize,
        max_tensor_elements: u64,
        max_live_tensor_handles: usize,
        max_read_bytes: u64,
    ) -> Self {
        Self {
            live_bytes: max_live_bytes,
            tensor_rank: max_tensor_rank,
            tensor_elements: max_tensor_elements,
            live_tensor_handles: max_live_tensor_handles,
            read_bytes: max_read_bytes,
            guest_memory_bytes: 4 * 1024 * 1024 * 1024,
            table_elements: 10_000,
            instances: 10_000,
        }
    }

    /// Overrides limits applied to WebAssembly store resources.
    #[must_use]
    pub const fn with_store_limits(
        mut self,
        guest_memory_bytes: usize,
        table_elements: usize,
        instances: usize,
    ) -> Self {
        self.guest_memory_bytes = guest_memory_bytes;
        self.table_elements = table_elements;
        self.instances = instances;
        self
    }
}

#[derive(Debug)]
struct BufferHandle;

/// Host-owned state behind a guest tensor resource.
#[derive(Clone, Debug)]
pub struct TensorEntry {
    tensor: Tensor,
    buffer: Arc<BufferHandle>,
}

/// Store state implementing the guest compute interface over a trusted backend.
pub struct Host<B: Backend> {
    backend: Arc<B>,
    table: ResourceTable,
    limits: Limits,
    store_limits: StoreLimits,
    live_bytes: Arc<AtomicU64>,
    live_handles: usize,
}

impl<B: Backend> Host<B> {
    fn new(backend: B, limits: Limits) -> Self {
        let store_limits = StoreLimitsBuilder::new()
            .memory_size(limits.guest_memory_bytes)
            .table_elements(limits.table_elements)
            .instances(limits.instances)
            .build();
        Self {
            backend: Arc::new(backend),
            table: ResourceTable::new(),
            limits,
            store_limits,
            live_bytes: Arc::new(AtomicU64::new(0)),
            live_handles: 0,
        }
    }

    /// Creates a Wasmtime store with the configured resource limiter installed.
    #[must_use]
    pub fn new_store(engine: &Engine, backend: B, limits: Limits) -> Store<Self>
    where
        B: Send + 'static,
    {
        let mut store = Store::new(engine, Self::new(backend, limits));
        store.limiter(|host| &mut host.store_limits);
        store
    }

    /// Allocates a contiguous tensor after enforcing all guest quotas.
    ///
    /// # Errors
    ///
    /// Returns a quota or backend error without creating a guest handle.
    pub fn alloc(
        &mut self,
        dtype: compute::Dtype,
        shape: &[u32],
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        let dtype = core_dtype(dtype);
        let byte_len = self.check_allocation(dtype, shape)?;
        let tensor = self.backend.alloc(dtype, shape).map_err(backend_error)?;
        let entry = TensorEntry {
            tensor: tensor.clone(),
            buffer: Arc::new(BufferHandle),
        };
        let resource = match self.table.push(entry) {
            Ok(resource) => resource,
            Err(error) => {
                self.backend.release(&tensor).map_err(backend_error)?;
                return Err(compute::Error::InvalidHandle(error.to_string()));
            }
        };
        self.live_bytes.fetch_add(byte_len, Ordering::AcqRel);
        self.live_handles += 1;
        Ok(resource)
    }

    /// Creates a validated metadata-only tensor view.
    ///
    /// # Errors
    ///
    /// Returns a layout, quota, backend, or invalid-handle error.
    pub fn view(
        &mut self,
        resource: &Resource<TensorEntry>,
        operation: compute::ViewOp,
    ) -> Result<Resource<TensorEntry>, compute::Error> {
        self.check_handle_quota()?;
        let entry = self.entry(resource)?.clone();
        let operation = core_view(operation).map_err(|error| layout_error(&error))?;
        validate_view(&entry.tensor, &operation).map_err(|error| layout_error(&error))?;
        let tensor = self
            .backend
            .view(&entry.tensor, operation)
            .map_err(backend_error)?;
        self.check_tensor_shape(tensor.layout().shape())?;
        let view = self
            .table
            .push(TensorEntry {
                tensor,
                buffer: Arc::clone(&entry.buffer),
            })
            .map_err(|error| compute::Error::InvalidHandle(error.to_string()))?;
        self.live_handles += 1;
        Ok(view)
    }

    /// Writes bytes to a contiguous tensor with an exact logical byte count.
    ///
    /// # Errors
    ///
    /// Returns a layout, backend, or invalid-handle error.
    pub fn write(
        &self,
        resource: &Resource<TensorEntry>,
        bytes: &[u8],
    ) -> Result<(), compute::Error> {
        let tensor = &self.entry(resource)?.tensor;
        if !tensor.layout().is_contiguous() {
            return Err(compute::Error::Layout(
                "writes require a contiguous tensor".to_owned(),
            ));
        }
        let expected = tensor
            .layout()
            .element_count()
            .checked_mul(tensor.layout().dtype().byte_size())
            .ok_or_else(|| compute::Error::Layout("tensor byte size overflowed".to_owned()))?;
        if u64::try_from(bytes.len()) != Ok(expected) {
            return Err(compute::Error::Layout(format!(
                "write has {} bytes but tensor requires {expected}",
                bytes.len()
            )));
        }
        self.backend.write(tensor, bytes).map_err(backend_error)
    }

    fn entry(&self, resource: &Resource<TensorEntry>) -> Result<&TensorEntry, compute::Error> {
        self.table
            .get(resource)
            .map_err(|error| compute::Error::InvalidHandle(error.to_string()))
    }

    fn check_allocation(&self, dtype: DType, shape: &[u32]) -> Result<u64, compute::Error> {
        self.check_handle_quota()?;
        let elements = self.check_tensor_shape(shape)?;
        let byte_len = elements
            .checked_mul(dtype.byte_size())
            .ok_or_else(|| quota("tensor byte size exceeds the guest limit"))?;
        if self
            .live_bytes
            .load(Ordering::Acquire)
            .checked_add(byte_len)
            .is_none_or(|bytes| bytes > self.limits.live_bytes)
        {
            return Err(quota("live tensor bytes exceed the guest limit"));
        }
        Ok(byte_len)
    }

    fn check_tensor_shape(&self, shape: &[u32]) -> Result<u64, compute::Error> {
        if shape.len() > self.limits.tensor_rank {
            return Err(quota("tensor rank exceeds the guest limit"));
        }
        let elements = shape
            .iter()
            .try_fold(1_u64, |count, &extent| count.checked_mul(u64::from(extent)));
        let Some(elements) = elements else {
            return Err(quota("tensor element count exceeds the guest limit"));
        };
        if elements > self.limits.tensor_elements {
            return Err(quota("tensor element count exceeds the guest limit"));
        }
        Ok(elements)
    }

    fn check_handle_quota(&self) -> Result<(), compute::Error> {
        if self.live_handles >= self.limits.live_tensor_handles {
            return Err(quota("live tensor handles exceed the guest limit"));
        }
        Ok(())
    }
}

fn core_dtype(dtype: compute::Dtype) -> DType {
    match dtype {
        compute::Dtype::F32 => DType::F32,
        compute::Dtype::F16 => DType::F16,
        compute::Dtype::Bf16 => DType::BF16,
        compute::Dtype::I32 => DType::I32,
        compute::Dtype::U32 => DType::U32,
    }
}

fn core_view(operation: compute::ViewOp) -> Result<ViewOp, LayoutError> {
    Ok(match operation {
        compute::ViewOp::Slice(specs) => ViewOp::Slice(
            specs
                .into_iter()
                .map(|spec| Slice::new(spec.start, spec.len, spec.step))
                .collect::<Result<_, _>>()?,
        ),
        compute::ViewOp::Reshape(shape) => ViewOp::Reshape(shape),
        compute::ViewOp::Permute(axes) => ViewOp::Permute(axes),
        compute::ViewOp::Broadcast(shape) => ViewOp::Broadcast(shape),
    })
}

fn validate_view(tensor: &Tensor, operation: &ViewOp) -> Result<(), LayoutError> {
    match operation {
        ViewOp::Slice(specs) => tensor.layout().slice(specs),
        ViewOp::Reshape(shape) => tensor.layout().reshape(shape),
        ViewOp::Permute(axes) => tensor.layout().permute(axes),
        ViewOp::Broadcast(shape) => tensor.layout().broadcast(shape),
    }
    .map(|_| ())
}

fn quota(message: &str) -> compute::Error {
    compute::Error::Quota(message.to_owned())
}

fn backend_error(error: BackendError) -> compute::Error {
    match error {
        BackendError::QuotaExceeded => quota("backend allocation quota exceeded"),
        BackendError::InvalidInput => {
            compute::Error::InvalidHandle("backend rejected the tensor handle".to_owned())
        }
        error => compute::Error::BackendExecution(error.to_string()),
    }
}

fn layout_error(error: &LayoutError) -> compute::Error {
    compute::Error::Layout(error.to_string())
}

#[cfg(test)]
mod tests {
    use forja_cpu::CpuBackend;
    use wasmtime::component::Resource;

    use super::{Host, Limits, bindings::l9o::gpu::compute};

    const GENEROUS: Limits = Limits::new(u64::MAX, 8, u64::MAX, 32, u64::MAX);

    #[test]
    fn refuses_each_quota_before_allocation() {
        let cases = [
            (Limits::new(3, 8, u64::MAX, 8, u64::MAX), vec![1]),
            (Limits::new(u64::MAX, 1, u64::MAX, 8, u64::MAX), vec![1, 1]),
            (Limits::new(u64::MAX, 8, 6, 8, u64::MAX), vec![7]),
            (Limits::new(u64::MAX, 8, u64::MAX, 0, u64::MAX), vec![1]),
        ];
        for (limits, shape) in cases {
            let mut host = Host::new(CpuBackend::new(), limits);
            assert!(matches!(
                host.alloc(compute::Dtype::F32, &shape),
                Err(compute::Error::Quota(_))
            ));
        }

        let mut host = Host::new(CpuBackend::new(), Limits::new(3, 8, 1, 1, u64::MAX));
        assert!(matches!(
            host.alloc(compute::Dtype::F32, &[1]),
            Err(compute::Error::Quota(_))
        ));
        host.limits.live_bytes = 4;
        let tensor = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        assert_eq!(
            host.table
                .get(&tensor)
                .unwrap()
                .tensor
                .buffer()
                .allocation(),
            0
        );
    }

    #[test]
    fn refuses_views_over_handle_and_element_limits() {
        let mut host = Host::new(CpuBackend::new(), Limits::new(16, 8, 1, 1, u64::MAX));
        let tensor = host.alloc(compute::Dtype::F32, &[1]).unwrap();
        assert!(matches!(
            host.view(
                &Resource::new_borrow(tensor.rep()),
                compute::ViewOp::Reshape(vec![1]),
            ),
            Err(compute::Error::Quota(_))
        ));

        host.limits.live_tensor_handles = 8;
        assert!(matches!(
            host.view(
                &Resource::new_borrow(tensor.rep()),
                compute::ViewOp::Broadcast(vec![4_000_000_000]),
            ),
            Err(compute::Error::Quota(_))
        ));
    }

    #[test]
    fn writes_require_contiguous_exactly_sized_bytes() {
        let mut host = Host::new(CpuBackend::new(), GENEROUS);
        let tensor = host.alloc(compute::Dtype::F32, &[7, 1024]).unwrap();
        assert!(matches!(
            host.write(&tensor, &[]),
            Err(compute::Error::Layout(_))
        ));

        let view = host
            .view(
                &Resource::new_borrow(tensor.rep()),
                compute::ViewOp::Permute(vec![1, 0]),
            )
            .unwrap();
        assert!(matches!(
            host.write(&view, &vec![0; 7 * 1024 * 4]),
            Err(compute::Error::Layout(_))
        ));
    }
}
