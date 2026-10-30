use super::{
    CPU_HOST, CpuBackend, DEVICE, DType, Error, NativeDevice, Result, Tensor, core_dtype,
    cpu_inputs,
};
#[cfg(all(feature = "native-metal", target_os = "macos"))]
use super::{METAL_HOST, MetalBackend, metal_inputs};

pub(crate) enum State {
    Cpu(forja_host::NativeState<CpuBackend>),
    #[cfg(all(feature = "native-metal", target_os = "macos"))]
    Metal(forja_host::NativeState<MetalBackend>),
}
fn state_error(error: impl std::fmt::Debug) -> Error {
    Error::new(format!("{error:?}"))
}
impl State {
    pub(crate) fn new(dtype: DType, shape: &[u32]) -> Result<Self> {
        match DEVICE.get() {
            NativeDevice::Cpu => CPU_HOST.with(|host| {
                host.state(core_dtype(dtype), shape)
                    .map(Self::Cpu)
                    .map_err(state_error)
            }),
            #[cfg(all(feature = "native-metal", target_os = "macos"))]
            NativeDevice::Metal => METAL_HOST.with(|host| {
                host.as_ref()
                    .map_err(Clone::clone)?
                    .state(core_dtype(dtype), shape)
                    .map(Self::Metal)
                    .map_err(state_error)
            }),
        }
    }
    pub(crate) fn delta_rule(&mut self, inputs: [&Tensor; 5], output: &Tensor) -> Result<()> {
        match self {
            Self::Cpu(state) => {
                let inputs: [_; 5] = cpu_inputs(&inputs)?
                    .try_into()
                    .map_err(|_| Error::new("invalid state input count"))?;
                let output = cpu_inputs(&[output])?;
                state.delta_rule(inputs, output[0]).map_err(state_error)
            }
            #[cfg(all(feature = "native-metal", target_os = "macos"))]
            Self::Metal(state) => {
                let inputs: [_; 5] = metal_inputs(&inputs)?
                    .try_into()
                    .map_err(|_| Error::new("invalid state input count"))?;
                let output = metal_inputs(&[output])?;
                state.delta_rule(inputs, output[0]).map_err(state_error)
            }
        }
    }
    pub(crate) fn causal_convolution(
        &mut self,
        input: &Tensor,
        weights: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        match self {
            Self::Cpu(state) => {
                let tensors = cpu_inputs(&[input, weights, output])?;
                state
                    .causal_convolution(tensors[0], tensors[1], tensors[2])
                    .map_err(state_error)
            }
            #[cfg(all(feature = "native-metal", target_os = "macos"))]
            Self::Metal(state) => {
                let tensors = metal_inputs(&[input, weights, output])?;
                state
                    .causal_convolution(tensors[0], tensors[1], tensors[2])
                    .map_err(state_error)
            }
        }
    }

    pub(crate) fn delta_rule_block(
        &mut self,
        inputs: [&Tensor; 5],
        output: &Tensor,
        variant: &str,
    ) -> Result<()> {
        match self {
            Self::Cpu(state) => {
                let inputs: [_; 5] = cpu_inputs(&inputs)?
                    .try_into()
                    .map_err(|_| Error::new("invalid state input count"))?;
                let output = cpu_inputs(&[output])?;
                state
                    .delta_rule_block(inputs, output[0], variant)
                    .map_err(state_error)
            }
            #[cfg(all(feature = "native-metal", target_os = "macos"))]
            Self::Metal(state) => {
                let inputs: [_; 5] = metal_inputs(&inputs)?
                    .try_into()
                    .map_err(|_| Error::new("invalid state input count"))?;
                let output = metal_inputs(&[output])?;
                state
                    .delta_rule_block(inputs, output[0], variant)
                    .map_err(state_error)
            }
        }
    }
}
