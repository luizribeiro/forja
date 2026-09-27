#![cfg_attr(
    all(not(target_family = "wasm"), not(feature = "native")),
    allow(
        dead_code,
        reason = "view payloads are consumed by the guest or native backend"
    )
)]

use crate::{Error, Result};

#[derive(Clone, Copy)]
pub(crate) enum DType {
    F32,
    F16,
    BF16,
    U32,
    I32,
}

pub(crate) struct Slice {
    pub(crate) start: u32,
    pub(crate) len: u32,
    pub(crate) step: u32,
}

pub(crate) enum View {
    Slice(Vec<Slice>),
    Reshape(Vec<u32>),
    Permute(Vec<u8>),
    Broadcast(Vec<u32>),
}

pub(crate) trait Backend {
    type Tensor;

    fn alloc(dtype: DType, shape: &[u32]) -> Result<Self::Tensor>;
    fn write(tensor: &Self::Tensor, bytes: &[u8]) -> Result<()>;
    fn view(tensor: &Self::Tensor, operation: View) -> Result<Self::Tensor>;
    fn read(tensor: &Self::Tensor) -> Result<Vec<u8>>;
}

#[cfg(target_family = "wasm")]
mod guest {
    #![allow(clippy::same_length_and_capacity)]

    wit_bindgen::generate!({
        path: "../../wit",
        world: "host",
    });

    use super::{Backend, DType, Error, Result, View};
    use l9o::gpu::compute;

    pub(crate) struct Guest;

    impl Backend for Guest {
        type Tensor = compute::Tensor;

        fn alloc(dtype: DType, shape: &[u32]) -> Result<Self::Tensor> {
            compute::Tensor::alloc(wit_dtype(dtype), shape).map_err(|error| guest_error(&error))
        }

        fn write(tensor: &Self::Tensor, bytes: &[u8]) -> Result<()> {
            tensor.write(bytes).map_err(|error| guest_error(&error))
        }

        fn view(tensor: &Self::Tensor, operation: View) -> Result<Self::Tensor> {
            tensor
                .view(&wit_view(operation))
                .map_err(|error| guest_error(&error))
        }

        fn read(tensor: &Self::Tensor) -> Result<Vec<u8>> {
            wit_bindgen::block_on(tensor.read()).map_err(|error| guest_error(&error))
        }
    }

    fn wit_dtype(dtype: DType) -> compute::Dtype {
        match dtype {
            DType::F32 => compute::Dtype::F32,
            DType::F16 => compute::Dtype::F16,
            DType::BF16 => compute::Dtype::Bf16,
            DType::U32 => compute::Dtype::U32,
            DType::I32 => compute::Dtype::I32,
        }
    }

    fn wit_view(operation: View) -> compute::ViewOp {
        match operation {
            View::Slice(slices) => compute::ViewOp::Slice(
                slices
                    .into_iter()
                    .map(|slice| compute::SliceSpec {
                        start: slice.start,
                        len: slice.len,
                        step: slice.step,
                    })
                    .collect(),
            ),
            View::Reshape(shape) => compute::ViewOp::Reshape(shape),
            View::Permute(axes) => compute::ViewOp::Permute(axes),
            View::Broadcast(shape) => compute::ViewOp::Broadcast(shape),
        }
    }

    fn guest_error(error: &compute::Error) -> Error {
        Error::new(format!("{error:?}"))
    }
}

#[cfg(all(not(target_family = "wasm"), not(feature = "native")))]
mod unavailable {
    use super::{Backend, DType, Error, Result, View};

    pub(crate) enum UnavailableTensor {}

    pub(crate) struct Unavailable;

    impl Backend for Unavailable {
        type Tensor = UnavailableTensor;

        fn alloc(_dtype: DType, _shape: &[u32]) -> Result<Self::Tensor> {
            Err(error())
        }

        fn write(_tensor: &Self::Tensor, _bytes: &[u8]) -> Result<()> {
            Err(error())
        }

        fn view(_tensor: &Self::Tensor, _operation: View) -> Result<Self::Tensor> {
            Err(error())
        }

        fn read(_tensor: &Self::Tensor) -> Result<Vec<u8>> {
            Err(error())
        }
    }

    fn error() -> Error {
        Error::new("forja-sdk requires WebAssembly or the native feature")
    }
}

#[cfg(feature = "native")]
mod native {
    use std::cell::Cell;

    use forja_core::{DType as CoreDType, Slice as CoreSlice, ViewOp};
    use forja_host::{NativeHost, NativeTensor};

    use super::{Backend, DType, Error, Result, View};
    use crate::NativeDevice;

    type CpuTensor = NativeTensor<forja_cpu::CpuBackend>;
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    type MetalTensor = NativeTensor<forja_metal::MetalBackend>;

    thread_local! {
        static DEVICE: Cell<NativeDevice> = const { Cell::new(NativeDevice::Cpu) };
        static CPU_HOST: NativeHost<forja_cpu::CpuBackend> =
            NativeHost::new(forja_cpu::CpuBackend::new());
        #[cfg(all(feature = "native-metal", target_os = "macos"))]
        static METAL_HOST: Result<NativeHost<forja_metal::MetalBackend>> =
            forja_metal::MetalBackend::new().map(NativeHost::new).map_err(error);
    }

    pub(crate) enum Tensor {
        Cpu(CpuTensor),
        #[cfg(all(feature = "native-metal", target_os = "macos"))]
        Metal(MetalTensor),
    }

    pub(crate) struct Native;

    impl Backend for Native {
        type Tensor = Tensor;

        fn alloc(dtype: DType, shape: &[u32]) -> Result<Self::Tensor> {
            match DEVICE.get() {
                NativeDevice::Cpu => CPU_HOST.with(|host| {
                    host.alloc(core_dtype(dtype), shape)
                        .map(Tensor::Cpu)
                        .map_err(error)
                }),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                NativeDevice::Metal => with_metal(|host| {
                    host.alloc(core_dtype(dtype), shape)
                        .map(Tensor::Metal)
                        .map_err(error)
                }),
            }
        }

        fn write(tensor: &Self::Tensor, bytes: &[u8]) -> Result<()> {
            match tensor {
                Tensor::Cpu(tensor) => {
                    CPU_HOST.with(|host| host.write(tensor, bytes).map_err(error))
                }
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                Tensor::Metal(tensor) => {
                    with_metal(|host| host.write(tensor, bytes).map_err(error))
                }
            }
        }

        fn view(tensor: &Self::Tensor, operation: View) -> Result<Self::Tensor> {
            let operation = core_view(operation)?;
            match tensor {
                Tensor::Cpu(tensor) => CPU_HOST
                    .with(|host| host.view(tensor, operation).map(Tensor::Cpu).map_err(error)),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                Tensor::Metal(tensor) => with_metal(|host| {
                    host.view(tensor, operation)
                        .map(Tensor::Metal)
                        .map_err(error)
                }),
            }
        }

        fn read(tensor: &Self::Tensor) -> Result<Vec<u8>> {
            match tensor {
                Tensor::Cpu(tensor) => CPU_HOST.with(|host| host.read(tensor).map_err(error)),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                Tensor::Metal(tensor) => with_metal(|host| host.read(tensor).map_err(error)),
            }
        }
    }

    pub(crate) fn set_device(device: NativeDevice) {
        DEVICE.set(device);
    }

    fn core_view(operation: View) -> Result<ViewOp> {
        match operation {
            View::Slice(slices) => Ok(ViewOp::Slice(
                slices
                    .into_iter()
                    .map(|slice| CoreSlice::new(slice.start, slice.len, slice.step))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|error| Error::new(error.to_string()))?,
            )),
            View::Reshape(shape) => Ok(ViewOp::Reshape(shape)),
            View::Permute(axes) => Ok(ViewOp::Permute(axes)),
            View::Broadcast(shape) => Ok(ViewOp::Broadcast(shape)),
        }
    }

    fn core_dtype(dtype: DType) -> CoreDType {
        match dtype {
            DType::F32 => CoreDType::F32,
            DType::F16 => CoreDType::F16,
            DType::BF16 => CoreDType::BF16,
            DType::U32 => CoreDType::U32,
            DType::I32 => CoreDType::I32,
        }
    }

    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    fn with_metal<T>(
        operation: impl FnOnce(&NativeHost<forja_metal::MetalBackend>) -> Result<T>,
    ) -> Result<T> {
        METAL_HOST.with(|host| match host {
            Ok(host) => operation(host),
            Err(error) => Err(error.clone()),
        })
    }

    fn error(error: forja_core::BackendError) -> Error {
        Error::new(error.to_string())
    }
}

#[cfg(all(not(feature = "native"), target_family = "wasm"))]
pub(crate) use guest::Guest as Active;
#[cfg(feature = "native")]
pub(crate) use native::Native as Active;
#[cfg(all(not(feature = "native"), not(target_family = "wasm")))]
pub(crate) use unavailable::Unavailable as Active;
pub(crate) type Handle = <Active as Backend>::Tensor;

pub(crate) fn alloc(dtype: u8, shape: &[u32]) -> Result<Handle> {
    let dtype = match dtype {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::U32,
        4 => DType::I32,
        _ => return Err(Error::new("unsupported tensor element type")),
    };
    Active::alloc(dtype, shape)
}

pub(crate) fn write(tensor: &Handle, bytes: &[u8]) -> Result<()> {
    Active::write(tensor, bytes)
}

pub(crate) fn view(tensor: &Handle, operation: View) -> Result<Handle> {
    Active::view(tensor, operation)
}

pub(crate) fn read(tensor: &Handle) -> Result<Vec<u8>> {
    Active::read(tensor)
}

#[cfg(feature = "native")]
pub(crate) fn set_native_device(device: crate::NativeDevice) {
    native::set_device(device);
}
