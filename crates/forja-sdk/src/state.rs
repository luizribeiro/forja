//! Eager updates of host-owned state that cannot be viewed as ordinary tensors.
use crate::{DType, Error, FloatElement, Result, Tensor, eval};

#[cfg(any(target_family = "wasm", feature = "native"))]
use crate::sys;

/// An opaque, zero-initialized private state plane.
/// Each update drains pending tensor operations and publishes only after successful execution.
/// State updates cannot be captured or replayed.
pub struct StatePlane {
    #[cfg(target_family = "wasm")]
    raw: sys::guest::compute::StatePlane,
    #[cfg(all(not(target_family = "wasm"), feature = "native"))]
    raw: sys::native::State,
}
impl StatePlane {
    /// Allocates a private state plane; dropping it releases its capability and backing.
    ///
    /// # Errors
    /// Returns invalid geometry, quota or allocation errors.
    pub fn new(dtype: DType, shape: &[u32]) -> Result<Self> {
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
            return Err(Error::new("state planes require a floating scalar type"));
        }
        eval()?;
        #[cfg(target_family = "wasm")]
        let raw = sys::guest::compute::StatePlane::create(sys::guest::wit_dtype(dtype), shape)
            .map_err(|error| Error::new(format!("{error:?}")))?;
        #[cfg(all(not(target_family = "wasm"), feature = "native"))]
        let raw = sys::native::State::new(dtype, shape)?;
        #[cfg(any(target_family = "wasm", feature = "native"))]
        return Ok(Self { raw });
        #[cfg(all(not(target_family = "wasm"), not(feature = "native")))]
        {
            let _ = (dtype, shape);
            Err(Error::new("state storage requires a host"))
        }
    }

    /// Advances a float32 recurrent plane using normalized Q/K and per-head decay/beta.
    ///
    /// # Errors
    /// Returns validation, quota or execution errors; failed work is never published.
    pub fn delta_rule<T: FloatElement>(
        &mut self,
        q: &Tensor<f32>,
        k: &Tensor<f32>,
        v: &Tensor<T>,
        decay: &Tensor<f32>,
        beta: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        let output = Tensor::<T>::zeros(v.shape())?;
        eval()?;
        let inputs = [
            q.handle(),
            k.handle(),
            v.handle(),
            decay.handle(),
            beta.handle(),
        ];
        #[cfg(target_family = "wasm")]
        self.raw
            .delta_rule(&inputs, output.handle())
            .map_err(|error| Error::new(format!("{error:?}")))?;
        #[cfg(all(not(target_family = "wasm"), feature = "native"))]
        self.raw.delta_rule(inputs, output.handle())?;
        #[cfg(all(not(target_family = "wasm"), not(feature = "native")))]
        {
            let _ = (inputs, output);
            return Err(Error::new("state storage requires a host"));
        }
        #[cfg(any(target_family = "wasm", feature = "native"))]
        Ok(output)
    }

    /// Applies raw causal depthwise convolution and carries its private input history.
    ///
    /// # Errors
    /// Returns validation, quota or execution errors; failed work is never published.
    pub fn causal_convolution<T: FloatElement>(
        &mut self,
        input: &Tensor<T>,
        weights: &Tensor<T>,
    ) -> Result<Tensor<T>> {
        let output = Tensor::<T>::zeros(input.shape())?;
        eval()?;
        #[cfg(target_family = "wasm")]
        self.raw
            .causal_convolution(input.handle(), weights.handle(), output.handle())
            .map_err(|error| Error::new(format!("{error:?}")))?;
        #[cfg(all(not(target_family = "wasm"), feature = "native"))]
        self.raw
            .causal_convolution(input.handle(), weights.handle(), output.handle())?;
        #[cfg(all(not(target_family = "wasm"), not(feature = "native")))]
        {
            let _ = (weights, output);
            return Err(Error::new("state storage requires a host"));
        }
        #[cfg(any(target_family = "wasm", feature = "native"))]
        Ok(output)
    }
}
