#![cfg_attr(
    all(not(target_family = "wasm"), not(feature = "native")),
    allow(
        dead_code,
        reason = "operation payloads are consumed by the guest or native backend"
    )
)]

use crate::program::{BinaryOp, ProgramKind, ReduceOp, UnaryOp, ValueType};
use crate::{Error, Result};

/// A scalar type accepted by kernel signatures.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DType {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
    /// Brain floating point with an eight-bit exponent.
    BF16,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
}

pub(crate) struct Slice {
    pub(crate) start: u32,
    pub(crate) len: u32,
    pub(crate) step: u32,
}

#[cfg_attr(
    not(target_family = "wasm"),
    allow(
        dead_code,
        reason = "affine fields are consumed by the WebAssembly host"
    )
)]
#[derive(Clone, Copy)]
pub(crate) struct Affine {
    pub(crate) param: Option<u8>,
    pub(crate) scale: u32,
    pub(crate) offset: u32,
}

#[cfg_attr(
    not(target_family = "wasm"),
    allow(
        dead_code,
        reason = "parameter slices are consumed by the WebAssembly host"
    )
)]
pub(crate) struct ParamSlice {
    pub(crate) start: Affine,
    pub(crate) len: Affine,
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
    Argmax,
    TopK {
        k: u32,
        normalize: bool,
    },
    Sample {
        position: Affine,
    },
    Rope(f32),
    QkvRopeCache {
        eps: f32,
        theta: f32,
    },
    Embed,
    QuantEmbed {
        bits: u8,
        group_size: u32,
    },
    Matmul,
    GatherMatmul,
    QuantMatmul {
        bits: u8,
        group_size: u32,
    },
    QuantizedRouter {
        group_size: u32,
        k: u32,
        normalize: bool,
    },
    GatherQuantMatmul {
        bits: u8,
        group_size: u32,
    },
    GatherQuantMatmulCombine {
        bits: u8,
        group_size: u32,
    },
    GatherQuantSiluMul {
        bits: u8,
        group_size: u32,
    },
    Sdpa {
        scale: f32,
        causal: bool,
        q_start: Affine,
    },
}

#[derive(Clone)]
pub(crate) struct Program {
    pub(crate) kind: ProgramKind,
    pub(crate) instructions: Vec<ProgramInst>,
    pub(crate) outputs: Vec<(u32, u32)>,
}

#[derive(Clone, Copy)]
pub(crate) enum ProgramInst {
    Input(u32),
    Constant(f32),
    Index(u8),
    Extent(u8),
    Unary(UnaryOp, u32),
    Binary(BinaryOp, u32, u32),
    Select(u32, u32, u32),
    Cast(ValueType, u32),
    Reduce(ReduceOp, u32),
}

pub(crate) trait Backend {
    type Tensor;
    type Commands;
    type Kernel;
    type Params;
    type Graph;

    fn alloc(dtype: DType, shape: &[u32]) -> Result<Self::Tensor>;
    fn write(tensor: &Self::Tensor, bytes: &[u8]) -> Result<()>;
    fn view(tensor: &Self::Tensor, operation: View) -> Result<Self::Tensor>;
    fn view_param(
        tensor: &Self::Tensor,
        params: &Self::Params,
        slices: &[ParamSlice],
    ) -> Result<Self::Tensor>;
    fn read(tensor: &Self::Tensor) -> Result<Vec<u8>>;
    fn command_list() -> Result<Self::Commands>;
    fn dispatch(
        commands: &mut Self::Commands,
        operation: Op,
        inputs: &[&Self::Tensor],
        output: &Self::Tensor,
    ) -> Result<()>;
    fn dispatch_many(
        commands: &mut Self::Commands,
        operation: Op,
        inputs: &[&Self::Tensor],
        outputs: &[&Self::Tensor],
    ) -> Result<()>;
    fn create_kernel(
        program: Program,
        rank: u8,
        inputs: &[DType],
        outputs: &[DType],
    ) -> Result<Self::Kernel>;
    fn dispatch_kernel(
        commands: &mut Self::Commands,
        kernel: &Self::Kernel,
        inputs: &[&Self::Tensor],
        outputs: &[&Self::Tensor],
    ) -> Result<()>;
    fn submit(commands: Self::Commands) -> Result<()>;
    fn params(ranges: &[(u32, u32)]) -> Result<Self::Params>;
    fn create_graph(commands: Self::Commands) -> Result<Self::Graph>;
    fn replay(graph: &Self::Graph, values: &[u32]) -> Result<()>;
}

#[cfg(target_family = "wasm")]
pub(crate) mod guest {
    #![allow(missing_docs, clippy::same_length_and_capacity)]

    wit_bindgen::generate!({
        path: "../../wit",
        world: "host",
    });

    use super::{Backend, DType, Error, Op, ParamSlice, Program, ProgramInst, Result, View};
    use crate::program::{BinaryOp, ProgramKind, ReduceOp, UnaryOp, ValueType};
    use compute::{
        Binop as WitBinOp, Redop as WitRedOp, Unop as WitUnOp, ValueType as WitValueType,
    };
    pub use l9o::gpu::compute;

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
        type Kernel = compute::Kernel;
        type Params = compute::Params;
        type Graph = compute::Graph;

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

        fn view_param(
            tensor: &Self::Tensor,
            params: &Self::Params,
            slices: &[ParamSlice],
        ) -> Result<Self::Tensor> {
            tensor
                .view_param(
                    params,
                    &slices
                        .iter()
                        .map(|slice| compute::ParamSlice {
                            start: wit_affine(slice.start),
                            len: wit_affine(slice.len),
                            step: slice.step,
                        })
                        .collect::<Vec<_>>(),
                )
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

        fn dispatch_many(
            commands: &mut Self::Commands,
            operation: Op,
            inputs: &[&Self::Tensor],
            outputs: &[&Self::Tensor],
        ) -> Result<()> {
            commands
                .dispatch_many(wit_op(operation), inputs, outputs)
                .map_err(|error| guest_error(&error))
        }

        fn create_kernel(
            program: Program,
            rank: u8,
            inputs: &[DType],
            outputs: &[DType],
        ) -> Result<Self::Kernel> {
            compute::Kernel::create(
                &wit_program(program),
                &compute::KernelSignature {
                    rank,
                    inputs: inputs.iter().copied().map(wit_dtype).collect(),
                    outputs: outputs.iter().copied().map(wit_dtype).collect(),
                    scalars: 0,
                },
            )
            .map_err(|error| guest_error(&error))
        }

        fn dispatch_kernel(
            commands: &mut Self::Commands,
            kernel: &Self::Kernel,
            inputs: &[&Self::Tensor],
            outputs: &[&Self::Tensor],
        ) -> Result<()> {
            commands
                .dispatch_kernel(kernel, inputs, outputs)
                .map_err(|error| guest_error(&error))
        }

        fn submit(commands: Self::Commands) -> Result<()> {
            wit_bindgen::block_on(compute::submit(commands))
                .map(|_| ())
                .map_err(|error| guest_error(&error))
        }

        fn params(ranges: &[(u32, u32)]) -> Result<Self::Params> {
            Ok(compute::Params::new(
                &ranges
                    .iter()
                    .map(|&(lo, hi)| compute::ParamRange { lo, hi })
                    .collect::<Vec<_>>(),
            ))
        }

        fn create_graph(commands: Self::Commands) -> Result<Self::Graph> {
            compute::Graph::create(commands).map_err(|error| guest_error(&error))
        }

        fn replay(graph: &Self::Graph, values: &[u32]) -> Result<()> {
            wit_bindgen::block_on(compute::replay(graph, values.to_vec()))
                .map(|_| ())
                .map_err(|error| guest_error(&error))
        }
    }

    fn wit_affine(affine: super::Affine) -> compute::Affine {
        compute::Affine {
            param: affine.param,
            scale: affine.scale,
            offset: affine.offset,
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
            Op::Argmax => compute::Op::Argmax,
            Op::TopK { k, normalize } => compute::Op::TopK(compute::TopKCfg { k, normalize }),
            Op::Sample { position } => compute::Op::Sample(compute::SampleCfg {
                position: wit_affine(position),
            }),
            Op::Rope(theta) => compute::Op::Rope(compute::RopeCfg { theta }),
            Op::QkvRopeCache { eps, theta } => {
                compute::Op::QkvRopeCache(compute::QkvRopeCacheCfg { eps, theta })
            }
            Op::Embed => compute::Op::Embed,
            Op::QuantEmbed { bits, group_size } => {
                compute::Op::QuantEmbed(compute::QuantMatmulCfg { bits, group_size })
            }
            Op::Matmul => compute::Op::Matmul,
            Op::GatherMatmul => compute::Op::GatherMatmul,
            Op::QuantMatmul { bits, group_size } => {
                compute::Op::QuantMatmul(compute::QuantMatmulCfg { bits, group_size })
            }
            Op::QuantizedRouter {
                group_size,
                k,
                normalize,
            } => compute::Op::QuantizedRouter(compute::QuantizedRouterCfg {
                group_size,
                k,
                normalize,
            }),
            Op::GatherQuantMatmul { bits, group_size } => {
                compute::Op::GatherQuantMatmul(compute::QuantMatmulCfg { bits, group_size })
            }
            Op::GatherQuantMatmulCombine { bits, group_size } => {
                compute::Op::GatherQuantMatmulCombine(compute::QuantMatmulCfg { bits, group_size })
            }
            Op::GatherQuantSiluMul { bits, group_size } => {
                compute::Op::GatherQuantSiluMul(compute::QuantMatmulCfg { bits, group_size })
            }
            Op::Sdpa {
                scale,
                causal,
                q_start,
            } => compute::Op::Sdpa(compute::SdpaCfg {
                scale,
                causal,
                q_start: wit_affine(q_start),
            }),
        }
    }

    fn wit_program(program: Program) -> compute::ProgramSource {
        compute::ProgramSource {
            kind: match program.kind {
                ProgramKind::Map => compute::ProgramKind::Map,
                ProgramKind::Row => compute::ProgramKind::Row,
            },
            insts: program.instructions.into_iter().map(wit_inst).collect(),
            outputs: program.outputs,
        }
    }

    fn wit_inst(instruction: ProgramInst) -> compute::Inst {
        match instruction {
            ProgramInst::Input(slot) => compute::Inst::Input(slot),
            ProgramInst::Constant(value) => compute::Inst::Const(value),
            ProgramInst::Index(axis) => compute::Inst::Index(axis),
            ProgramInst::Extent(axis) => compute::Inst::Extent(axis),
            ProgramInst::Unary(op, value) => compute::Inst::Unary((wit_unop(op), value)),
            ProgramInst::Binary(op, left, right) => {
                compute::Inst::Binary((wit_binop(op), left, right))
            }
            ProgramInst::Select(condition, accepted, rejected) => {
                compute::Inst::Select((condition, accepted, rejected))
            }
            ProgramInst::Cast(to, value) => compute::Inst::Cast((wit_value_type(to), value)),
            ProgramInst::Reduce(op, value) => compute::Inst::Reduce((wit_redop(op), value)),
        }
    }

    forja_program_conversions::program_op_conversions! {
        fn wit_unop(UnaryOp => WitUnOp);
        fn wit_binop(BinaryOp => WitBinOp);
        fn wit_redop(ReduceOp => WitRedOp);
    }

    const fn wit_value_type(value_type: ValueType) -> WitValueType {
        match value_type {
            ValueType::F32 => WitValueType::F32,
            ValueType::U32 => WitValueType::U32,
            ValueType::Bool => WitValueType::Bool,
        }
    }

    fn guest_error(error: &compute::Error) -> Error {
        Error::new(format!("{error:?}"))
    }
}

#[cfg(all(not(target_family = "wasm"), not(feature = "native")))]
pub(crate) mod unavailable {
    use super::{Backend, DType, Error, Op, ParamSlice, Program, Result, View};

    pub(crate) enum UnavailableTensor {}
    pub(crate) enum UnavailableCommands {}
    pub(crate) enum UnavailableKernel {}
    pub(crate) enum UnavailableParams {}
    pub(crate) enum UnavailableGraph {}

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
        type Kernel = UnavailableKernel;
        type Params = UnavailableParams;
        type Graph = UnavailableGraph;

        fn alloc(_dtype: DType, _shape: &[u32]) -> Result<Self::Tensor> {
            Err(error())
        }

        fn write(_tensor: &Self::Tensor, _bytes: &[u8]) -> Result<()> {
            Err(error())
        }

        fn view(_tensor: &Self::Tensor, _operation: View) -> Result<Self::Tensor> {
            Err(error())
        }

        fn view_param(
            _tensor: &Self::Tensor,
            _params: &Self::Params,
            _slices: &[ParamSlice],
        ) -> Result<Self::Tensor> {
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

        fn dispatch_many(
            _commands: &mut Self::Commands,
            _operation: Op,
            _inputs: &[&Self::Tensor],
            _outputs: &[&Self::Tensor],
        ) -> Result<()> {
            Err(error())
        }

        fn create_kernel(
            _program: Program,
            _rank: u8,
            _inputs: &[DType],
            _outputs: &[DType],
        ) -> Result<Self::Kernel> {
            Err(error())
        }

        fn dispatch_kernel(
            _commands: &mut Self::Commands,
            _kernel: &Self::Kernel,
            _inputs: &[&Self::Tensor],
            _outputs: &[&Self::Tensor],
        ) -> Result<()> {
            Err(error())
        }

        fn submit(_commands: Self::Commands) -> Result<()> {
            Err(error())
        }

        fn params(_ranges: &[(u32, u32)]) -> Result<Self::Params> {
            Err(error())
        }

        fn create_graph(_commands: Self::Commands) -> Result<Self::Graph> {
            Err(error())
        }

        fn replay(_graph: &Self::Graph, _values: &[u32]) -> Result<()> {
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

    use forja_core::{
        DType as CoreDType, Op as CoreOp, Slice as CoreSlice, ViewOp,
        program::{
            BinOp, Inst, Program as CoreProgram, ProgramKind as CoreProgramKind, RedOp, UnOp,
            ValueType as CoreValueType,
        },
    };
    use forja_host::{
        NativeCommandList, NativeHost, NativeKernel, NativeTensor, Safetensors, WeightSource as _,
    };

    use super::{Backend, DType, Error, Op, ParamSlice, Program, ProgramInst, Result, View};
    use crate::NativeDevice;
    use crate::program::{BinaryOp, ProgramKind, ReduceOp, UnaryOp, ValueType};

    type CpuBackend = forja_cpu::CpuBackend;
    type CpuTensor = NativeTensor<CpuBackend>;
    type CpuCommands = NativeCommandList<CpuBackend>;
    type CpuKernel = NativeKernel<CpuBackend>;
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    type MetalBackend = forja_metal::MetalBackend;
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    type MetalTensor = NativeTensor<MetalBackend>;
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    type MetalCommands = NativeCommandList<MetalBackend>;
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    type MetalKernel = NativeKernel<MetalBackend>;

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

    pub(crate) enum Kernel {
        Cpu(CpuKernel),
        #[cfg(all(feature = "native-metal", target_os = "macos"))]
        Metal(MetalKernel),
    }

    pub(crate) enum Params {}
    pub(crate) enum Graph {}

    pub(crate) struct Native;

    pub(crate) struct WeightSource {
        shards: Vec<(Safetensors, forja_core::MappedRegion)>,
    }

    impl WeightSource {
        pub(crate) fn open(path: &std::path::Path) -> Result<Self> {
            let sources =
                Safetensors::open_all(path).map_err(|error| Error::new(error.to_string()))?;
            let shards = sources
                .into_iter()
                .map(|source| {
                    let region = source
                        .mapped_region()
                        .map_err(|error| Error::new(error.to_string()))?;
                    Ok((source, region))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Self { shards })
        }

        pub(crate) fn tensor(&self, name: &str, dtype: DType, shape: &[u32]) -> Result<Tensor> {
            let (tensor, region) = self
                .shards
                .iter()
                .find_map(|(source, region)| {
                    source
                        .tensors()
                        .iter()
                        .find(|tensor| tensor.name() == name)
                        .map(|tensor| (tensor, region))
                })
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
            let bytes = region
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
        type Kernel = Kernel;
        type Params = Params;
        type Graph = Graph;

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

        fn view_param(
            _tensor: &Self::Tensor,
            _params: &Self::Params,
            _slices: &[ParamSlice],
        ) -> Result<Self::Tensor> {
            Err(Error::new("graph capture requires a WebAssembly host"))
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
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                _ => Err(Error::new("native tensors belong to different backends")),
            }
        }

        fn dispatch_many(
            commands: &mut Self::Commands,
            operation: Op,
            inputs: &[&Self::Tensor],
            outputs: &[&Self::Tensor],
        ) -> Result<()> {
            match commands {
                Commands::Cpu(commands) => commands
                    .dispatch_many(
                        core_op(operation),
                        &cpu_inputs(inputs)?,
                        &cpu_inputs(outputs)?,
                    )
                    .map_err(error),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                Commands::Metal(commands) => commands
                    .dispatch_many(
                        core_op(operation),
                        &metal_inputs(inputs)?,
                        &metal_inputs(outputs)?,
                    )
                    .map_err(error),
            }
        }

        fn create_kernel(
            program: Program,
            rank: u8,
            inputs: &[DType],
            outputs: &[DType],
        ) -> Result<Self::Kernel> {
            let program = core_program(program)?;
            let signature = forja_core::program::KernelSignature::new(
                rank,
                inputs.iter().copied().map(core_dtype).collect(),
                outputs.iter().copied().map(core_dtype).collect(),
                0,
            );
            match DEVICE.get() {
                NativeDevice::Cpu => CPU_HOST.with(|host| {
                    host.prepare_program(program, signature)
                        .map(Kernel::Cpu)
                        .map_err(error)
                }),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                NativeDevice::Metal => with_metal(|host| {
                    host.prepare_program(program, signature)
                        .map(Kernel::Metal)
                        .map_err(error)
                }),
            }
        }

        fn dispatch_kernel(
            commands: &mut Self::Commands,
            kernel: &Self::Kernel,
            inputs: &[&Self::Tensor],
            outputs: &[&Self::Tensor],
        ) -> Result<()> {
            match (commands, kernel) {
                (Commands::Cpu(commands), Kernel::Cpu(kernel)) => commands
                    .dispatch_kernel(kernel, &cpu_inputs(inputs)?, &cpu_inputs(outputs)?)
                    .map_err(error),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                (Commands::Metal(commands), Kernel::Metal(kernel)) => commands
                    .dispatch_kernel(kernel, &metal_inputs(inputs)?, &metal_inputs(outputs)?)
                    .map_err(error),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                _ => Err(Error::new("native resources belong to different backends")),
            }
        }

        fn submit(commands: Self::Commands) -> Result<()> {
            match commands {
                Commands::Cpu(commands) => commands.submit().map(|_| ()).map_err(error),
                #[cfg(all(feature = "native-metal", target_os = "macos"))]
                Commands::Metal(commands) => commands.submit().map(|_| ()).map_err(error),
            }
        }

        fn params(_ranges: &[(u32, u32)]) -> Result<Self::Params> {
            Err(Error::new("graph capture requires a WebAssembly host"))
        }

        fn create_graph(_commands: Self::Commands) -> Result<Self::Graph> {
            Err(Error::new("graph capture requires a WebAssembly host"))
        }

        fn replay(_graph: &Self::Graph, _values: &[u32]) -> Result<()> {
            Err(Error::new("graph capture requires a WebAssembly host"))
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
            Op::Argmax => CoreOp::Argmax,
            Op::TopK { k, normalize } => CoreOp::TopK { k, normalize },
            Op::Sample { position } => CoreOp::Sample {
                position: position.offset,
            },
            Op::Rope(theta) => CoreOp::Rope { theta },
            Op::QkvRopeCache { eps, theta } => CoreOp::QkvRopeCache { eps, theta },
            Op::Embed => CoreOp::Embed,
            Op::QuantEmbed { bits, group_size } => CoreOp::QuantEmbed { bits, group_size },
            Op::Matmul => CoreOp::Matmul,
            Op::GatherMatmul => CoreOp::GatherMatmul,
            Op::QuantMatmul { bits, group_size } => CoreOp::QuantMatmul { bits, group_size },
            Op::QuantizedRouter {
                group_size,
                k,
                normalize,
            } => CoreOp::QuantizedRouter {
                group_size,
                k,
                normalize,
            },
            Op::GatherQuantMatmul { bits, group_size } => {
                CoreOp::GatherQuantMatmul { bits, group_size }
            }
            Op::GatherQuantMatmulCombine { bits, group_size } => {
                CoreOp::GatherQuantMatmulCombine { bits, group_size }
            }
            Op::GatherQuantSiluMul { bits, group_size } => {
                CoreOp::GatherQuantSiluMul { bits, group_size }
            }
            Op::Sdpa {
                scale,
                causal,
                q_start,
            } => CoreOp::Sdpa {
                scale,
                causal,
                q_start: q_start.offset,
            },
        }
    }

    pub(crate) fn core_program(program: Program) -> Result<forja_core::program::ValidatedProgram> {
        CoreProgram {
            kind: match program.kind {
                ProgramKind::Map => CoreProgramKind::Map,
                ProgramKind::Row => CoreProgramKind::Row,
            },
            insts: program.instructions.into_iter().map(core_inst).collect(),
            outputs: program.outputs,
        }
        .validate()
        .map_err(|error| Error::new(error.to_string()))
    }

    fn core_inst(instruction: ProgramInst) -> Inst {
        match instruction {
            ProgramInst::Input(slot) => Inst::Input(slot),
            ProgramInst::Constant(value) => Inst::Const(value),
            ProgramInst::Index(axis) => Inst::Index(axis),
            ProgramInst::Extent(axis) => Inst::Extent(axis),
            ProgramInst::Unary(op, value) => Inst::Unary(core_unop(op), value),
            ProgramInst::Binary(op, left, right) => Inst::Binary(core_binop(op), left, right),
            ProgramInst::Select(condition, accepted, rejected) => {
                Inst::Select(condition, accepted, rejected)
            }
            ProgramInst::Cast(to, value) => Inst::Cast(core_value_type(to), value),
            ProgramInst::Reduce(op, value) => Inst::Reduce(core_redop(op), value),
        }
    }

    forja_program_conversions::program_op_conversions! {
        fn core_unop(UnaryOp => UnOp);
        fn core_binop(BinaryOp => BinOp);
        fn core_redop(ReduceOp => RedOp);
    }

    const fn core_value_type(value_type: ValueType) -> CoreValueType {
        match value_type {
            ValueType::F32 => CoreValueType::F32,
            ValueType::U32 => CoreValueType::U32,
            ValueType::Bool => CoreValueType::Bool,
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
pub(crate) type Kernel = <Active as Backend>::Kernel;
pub(crate) type Params = <Active as Backend>::Params;
pub(crate) type Graph = <Active as Backend>::Graph;

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

pub(crate) fn view_param(
    tensor: &Handle,
    params: &Params,
    slices: &[ParamSlice],
) -> Result<Handle> {
    Active::view_param(tensor, params, slices)
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

pub(crate) fn dispatch_many(
    commands: &mut Commands,
    operation: Op,
    inputs: &[&Handle],
    outputs: &[&Handle],
) -> Result<()> {
    Active::dispatch_many(commands, operation, inputs, outputs)
}

pub(crate) fn create_kernel(
    program: Program,
    rank: u8,
    inputs: &[DType],
    outputs: &[DType],
) -> Result<Kernel> {
    Active::create_kernel(program, rank, inputs, outputs)
}

pub(crate) fn dispatch_kernel(
    commands: &mut Commands,
    kernel: &Kernel,
    inputs: &[&Handle],
    outputs: &[&Handle],
) -> Result<()> {
    Active::dispatch_kernel(commands, kernel, inputs, outputs)
}

pub(crate) fn submit(commands: Commands) -> Result<()> {
    Active::submit(commands)
}

pub(crate) fn params(ranges: &[(u32, u32)]) -> Result<Params> {
    Active::params(ranges)
}

pub(crate) fn create_graph(commands: Commands) -> Result<Graph> {
    Active::create_graph(commands)
}

pub(crate) fn replay(graph: &Graph, values: &[u32]) -> Result<()> {
    Active::replay(graph, values)
}

#[cfg(feature = "native")]
pub(crate) fn set_native_device(device: crate::NativeDevice) {
    native::set_device(device);
}
