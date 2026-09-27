use std::marker::PhantomData;
#[cfg(feature = "native")]
use std::rc::Rc;

use crate::{Element, Result, Tensor, sys};

/// A named collection of model tensors.
pub struct Weights<'a> {
    prefix: String,
    #[cfg(all(target_family = "wasm", not(feature = "native")))]
    source: sys::guest::WeightSource<'a>,
    #[cfg(feature = "native")]
    source: Rc<sys::native::WeightSource>,
    #[cfg(all(not(target_family = "wasm"), not(feature = "native")))]
    source: sys::unavailable::WeightSource,
    lifetime: PhantomData<&'a ()>,
}

impl Weights<'_> {
    /// Opens a safetensors file for native engine development.
    ///
    /// # Errors
    ///
    /// Returns an error when the file or any tensor metadata is invalid.
    #[cfg(feature = "native")]
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Weights<'static>> {
        Ok(Weights {
            prefix: String::new(),
            source: Rc::new(sys::native::WeightSource::open(path.as_ref())?),
            lifetime: PhantomData,
        })
    }

    /// Creates a namespace below this weight collection.
    #[must_use]
    pub fn scoped(&self, name: impl AsRef<str>) -> Self {
        let name = name.as_ref();
        let prefix = if self.prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{}.{name}", self.prefix)
        };
        Self {
            prefix,
            #[cfg(all(target_family = "wasm", not(feature = "native")))]
            source: sys::guest::WeightSource::new(self.source.raw()),
            #[cfg(feature = "native")]
            source: Rc::clone(&self.source),
            #[cfg(all(not(target_family = "wasm"), not(feature = "native")))]
            source: sys::unavailable::WeightSource,
            lifetime: PhantomData,
        }
    }

    /// Loads a tensor with an exact scalar type and shape.
    ///
    /// # Errors
    ///
    /// Returns an error naming the tensor when it is absent or has unexpected metadata.
    pub fn tensor<T: Element>(&self, name: &str, shape: &[u32]) -> Result<Tensor<T>> {
        let name = if self.prefix.is_empty() {
            name.to_owned()
        } else if name.is_empty() {
            self.prefix.clone()
        } else {
            format!("{}.{name}", self.prefix)
        };
        let handle = self.source.tensor(&name, sys::dtype(T::DTYPE)?, shape)?;
        Ok(Tensor::from_handle(handle, shape.to_vec()))
    }
}

#[cfg(all(target_family = "wasm", not(feature = "native")))]
impl<'a> Weights<'a> {
    #[doc(hidden)]
    pub fn from_guest(weights: &'a sys::guest::RawWeights) -> Self {
        Self {
            prefix: String::new(),
            source: sys::guest::WeightSource::new(weights),
            lifetime: PhantomData,
        }
    }
}

/// Constructs a model value from a named weight collection and configuration.
pub trait Load<C = ()>: Sized {
    /// Loads and validates every tensor needed by this value.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is missing or has incompatible metadata.
    fn load(weights: &Weights<'_>, config: &C) -> Result<Self>;
}
