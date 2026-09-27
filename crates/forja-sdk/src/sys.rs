#![cfg_attr(
    all(not(target_family = "wasm"), not(feature = "native")),
    allow(
        dead_code,
        reason = "operation payloads are consumed by the guest or native backend"
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

#[derive(Clone, Copy)]
pub(crate) enum Op {
    Copy,
    Add,
    SiluMul,
    RmsNorm(f32),
    Softmax,
    Rope(f32),
    Embed,
    Matmul,
    Sdpa {
        scale: f32,
        causal: bool,
        q_start: u32,
    },
}

pub(crate) trait Backend {
    type Tensor;
    type Commands;

    fn alloc(dtype: DType, shape: &[u32]) -> Result<Self::Tensor>;
    fn write(tensor: &Self::Tensor, bytes: &[u8]) -> Result<()>;
    fn view(tensor: &Self::Tensor, operation: View) -> Result<Self::Tensor>;
    fn read(tensor: &Self::Tensor) -> Result<Vec<u8>>;
    fn command_list() -> Result<Self::Commands>;
    fn dispatch(
        commands: &mut Self::Commands,
        operation: Op,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
    ) -> Result<()>;
    fn submit(commands: Self::Commands) -> Result<()>;
}

#[cfg(target_family = "wasm")]
pub(crate) mod guest {
    #![allow(clippy::same_length_and_capacity)]

    wit_bindgen::generate!({
        path: "../../wit",
        world: "host",
    });

    use super::{Backend, DType, Error, Op, Result, View};
    use l9o::gpu::compute;

    pub(crate) struct Guest;

    /// A borrowed weight resource supplied to an engine export.
    pub struct WeightSource<'a>(&'a compute::Weights);

    impl<'a> WeightSource<'a> {
        pub(crate) const fn new(weights: &'a compute::Weights) -> Self {
            Self(weights)
        }

        pub(crate) const fn raw(&self) -> &'a compute::Weights {
            self.0
        }

        pub(crate) fn tensor(
            &self,
            name: &str,
            dtype: DType,
            shape: &[u32],
        ) -> Result<compute::Tensor> {
            let info = self.0.info(name).map_err(|error| guest_error(&error))?;
            if info.dtype != wit_dtype(dtype) || info.shape != shape {
                return Err(Error::new(format!(
                    "weight {name:?} has {:?} {:?}, expected {:?} {shape:?}",
                    info.dtype,
                    info.shape,
                    wit_dtype(dtype),
                )));
            }
            self.0.tensor(name).map_err(|error| guest_error(&error))
        }
    }

    pub use compute::Weights as RawWeights;

    impl Backend for Guest {
        type Tensor = compute::Tensor;
        type Commands = compute::CommandList;

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

        fn command_list() -> Result<Self::Commands> {
            Ok(compute::CommandList::new())
        }

        fn dispatch(
            commands: &mut Self::Commands,
            operation: Op,
            inputs: &[&Self::Tensor],
            output: &Self::Tensor,
        ) -> Result<()> {
            commands
                .dispatch(wit_op(operation), inputs, output)
                .map_err(|error| guest_error(&error))
        }

        fn submit(commands: Self::Commands) -> Result<()> {
            wit_bindgen::block_on(compute::submit(commands))
                .map(|_| ())
                .map_err(|error| guest_error(&error))
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

    fn wit_op(operation: Op) -> compute::Op {
        match operation {
            Op::Copy => compute::Op::Copy,
            Op::Add => compute::Op::Add,
            Op::SiluMul => compute::Op::SiluMul,
            Op::RmsNorm(eps) => compute::Op::RmsNorm(eps),
            Op::Softmax => compute::Op::Softmax,
            Op::Rope(theta) => compute::Op::Rope(compute::RopeCfg { theta }),
            Op::Embed => compute::Op::Embed,
            Op::Matmul => compute::Op::Matmul,
            Op::Sdpa {
                scale,
                causal,
                q_start,
            } => compute::Op::Sdpa(compute::SdpaCfg {
                scale,
                causal,
                q_start,
            }),
        }
    }

    fn guest_error(error: &compute::Error) -> Error {
        Error::new(format!("{error:?}"))
    }
}

#[cfg(all(not(target_family = "wasm"), not(feature = "native")))]
pub(crate) mod unavailable {
    use super::{Backend, DType, Error, Op, Result, View};

    pub(crate) enum UnavailableTensor {}
    pub(crate) enum UnavailableCommands {}

    pub(crate) struct Unavailable;
    pub(crate) struct WeightSource;

    impl WeightSource {
        pub(crate) fn tensor(
            &self,
            _name: &str,
            _dtype: DType,
            _shape: &[u32],
        ) -> Result<UnavailableTensor> {
            Err(error())
        }
    }

    impl Backend for Unavailable {
        type Tensor = UnavailableTensor;
        type Commands = UnavailableCommands;

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

        fn command_list() -> Result<Self::Commands> {
            Err(error())
        }

        fn dispatch(
            _commands: &mut Self::Commands,
            _operation: Op,
            _inputs: &[&Self::Tensor],
            _output: &Self::Tensor,
        ) -> Result<()> {
            Err(error())
        }

        fn submit(_commands: Self::Commands) -> Result<()> {
            Err(error())
        }
    }

    fn error() -> Error {
        Error::new("forja-sdk requires WebAssembly or the native feature")
    }
}

#[cfg(feature = "native")]
pub(crate) mod native {
    use std::cell::Cell;

    use forja_core::{DType as CoreDType, Op as CoreOp, Slice as CoreSlice, ViewOp};
    use forja_host::{NativeCommandList, NativeHost, NativeTensor, Safetensors, WeightSource as _};

    use super::{Backend, DType, Error, Op, Result, View};
    use crate::NativeDevice;

    type CpuBackend = forja_cpu::CpuBackend;
    type CpuTensor = NativeTensor<CpuBackend>;
    type CpuCommands = NativeCommandList<CpuBackend>;
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    type MetalBackend = forja_metal::MetalBackend;
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    type MetalTensor = NativeTensor<MetalBackend>;
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    type MetalCommands = NativeCommandList<MetalBackend>;

    thread_local! {
        static DEVICE: Cell<NativeDevice> = const { Cell::new(NativeDevice::Cpu) };
        static CPU_HOST: NativeHost<CpuBackend> = NativeHost::new(CpuBackend::new());
        #[cfg(all(feature = "native-metal", target_os = "macos"))]
        static METAL_HOST: Result<NativeHost<MetalBackend>> =
            MetalBackend::new().map(NativeHost::new).map_err(error);
    }

    pub(crate) enum Tensor {
        Cpu(CpuTensor),
        #[cfg(all(feature = "native-metal", target_os = "macos"))]
        Metal(MetalTensor),
    }

    pub(crate) enum Commands {
        Cpu(CpuCommands),
        #[cfg(all(feature = "native-metal", target_os = "macos"))]
        Metal(MetalCommands),
    }

    pub(crate) struct Native;

    pub(crate) struct WeightSource {
        source: Safetensors,
        region: forja_core::MappedRegion,
    }

    impl WeightSource {
        pub(crate) fn open(path: &std::path::Path) -> Result<Self> {
            let source = Safetensors::open(path).map_err(|error| Error::new(error.to_string()))?;
            let region = source
                .mapped_region()
                .map_err(|error| Error::new(error.to_string()))?;
            Ok(Self { source, region })
        }

        pub(crate) fn tensor(&self, name: &str, dtype: DType, shape: &[u32]) -> Result<Tensor> {
            let tensor = self
                .source
                .tensors()
                .iter()
                .find(|tensor| tensor.name() == name)
                .ok_or_else(|| Error::new(format!("weight {name:?} is missing")))?;
            if tensor.dtype() != core_dtype(dtype) || tensor.shape() != shape {
                return Err(Error::new(format!(
                    "weight {name:?} has {:?} {:?}, expected {:?} {shape:?}",
                    tensor.dtype(),
                    tensor.shape(),
                    core_dtype(dtype),
                )));
            }
            let start = usize::try_from(tensor.byte_offset())
                .map_err(|_| Error::new(format!("weight {name:?} offset is too large")))?;
            let len = usize::try_from(tensor.byte_len())
                .map_err(|_| Error::new(format!("weight {name:?} length is too large")))?;
            let end = start
                .checked_add(len)
                .ok_or_else(|| Error::new(format!("weight {name:?} range overflowed")))?;
            let bytes = self
                .region
                .bytes()
                .get(start..end)
                .ok_or_else(|| Error::new(format!("weight {name:?} range is invalid")))?;
            let handle = Native::alloc(dtype, shape)?;
            Native::write(&handle, bytes)?;
            Ok(handle)
        }
    }

    impl Backend for Native {
        type Tensor = Tensor;
        type Commands = Commands;

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

        fn command_list() -> Result<Self::Commands> {
            match DEVICE.get() {
                NativeDevice::Cpu => CPU_HOST.with(|host| Ok(Commands::Cpu(host.command_list()))),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                NativeDevice::Metal => with_metal(|host| Ok(Commands::Metal(host.command_list()))),
            }
        }

        fn dispatch(
            commands: &mut Self::Commands,
            operation: Op,
            inputs: &[&Self::Tensor],
            output: &Self::Tensor,
        ) -> Result<()> {
            match (commands, output) {
                (Commands::Cpu(commands), Tensor::Cpu(output)) => commands
                    .dispatch(core_op(operation), &cpu_inputs(inputs)?, output)
                    .map_err(error),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                (Commands::Metal(commands), Tensor::Metal(output)) => commands
                    .dispatch(core_op(operation), &metal_inputs(inputs)?, output)
                    .map_err(error),
                _ => Err(Error::new("native tensors belong to different backends")),
            }
        }

        fn submit(commands: Self::Commands) -> Result<()> {
            match commands {
                Commands::Cpu(commands) => commands.submit().map(|_| ()).map_err(error),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                Commands::Metal(commands) => commands.submit().map(|_| ()).map_err(error),
            }
        }
    }

    pub(crate) fn set_device(device: NativeDevice) {
        DEVICE.set(device);
    }

    fn cpu_inputs<'a>(inputs: &[&'a Tensor]) -> Result<Vec<&'a CpuTensor>> {
        inputs
            .iter()
            .map(|tensor| match tensor {
                Tensor::Cpu(tensor) => Ok(tensor),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                Tensor::Metal(_) => Err(Error::new("native tensors belong to different backends")),
            })
            .collect()
    }

    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    fn metal_inputs<'a>(inputs: &[&'a Tensor]) -> Result<Vec<&'a MetalTensor>> {
        inputs
            .iter()
            .map(|tensor| match tensor {
                Tensor::Metal(tensor) => Ok(tensor),
                Tensor::Cpu(_) => Err(Error::new("native tensors belong to different backends")),
            })
            .collect()
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

    fn core_op(operation: Op) -> CoreOp {
        match operation {
            Op::Copy => CoreOp::Copy,
            Op::Add => CoreOp::Add,
            Op::SiluMul => CoreOp::SiluMul,
            Op::RmsNorm(eps) => CoreOp::RmsNorm { eps },
            Op::Softmax => CoreOp::Softmax,
            Op::Rope(theta) => CoreOp::Rope { theta },
            Op::Embed => CoreOp::Embed,
            Op::Matmul => CoreOp::Matmul,
            Op::Sdpa {
                scale,
                causal,
                q_start,
            } => CoreOp::Sdpa {
                scale,
                causal,
                q_start,
            },
        }
    }

    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    fn with_metal<T>(operation: impl FnOnce(&NativeHost<MetalBackend>) -> Result<T>) -> Result<T> {
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
pub(crate) type Commands = <Active as Backend>::Commands;

pub(crate) fn dtype(dtype: u8) -> Result<DType> {
    match dtype {
        0 => Ok(DType::F32),
        1 => Ok(DType::F16),
        2 => Ok(DType::BF16),
        3 => Ok(DType::U32),
        4 => Ok(DType::I32),
        _ => Err(Error::new("unsupported tensor element type")),
    }
}

pub(crate) fn alloc(dtype: u8, shape: &[u32]) -> Result<Handle> {
    Active::alloc(self::dtype(dtype)?, shape)
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

pub(crate) fn command_list() -> Result<Commands> {
    Active::command_list()
}

pub(crate) fn dispatch(
    commands: &mut Commands,
    operation: Op,
    inputs: &[&Handle],
    output: &Handle,
) -> Result<()> {
    Active::dispatch(commands, operation, inputs, output)
}

pub(crate) fn submit(commands: Commands) -> Result<()> {
    Active::submit(commands)
}

#[cfg(feature = "native")]
pub(crate) fn set_native_device(device: crate::NativeDevice) {
    native::set_device(device);
}
