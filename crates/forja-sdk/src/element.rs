use half::{bf16, f16};

use crate::{Error, Result};

mod sealed {
    pub trait Sealed {}
}

/// A scalar type supported by Forja tensors.
pub trait Element: sealed::Sealed + Copy + 'static {
    #[doc(hidden)]
    const DTYPE: u8;

    #[doc(hidden)]
    fn encode(values: &[Self]) -> Vec<u8>;

    #[doc(hidden)]
    fn decode(bytes: &[u8]) -> Result<Vec<Self>>;
}

macro_rules! element {
    ($type:ty, $dtype:literal, $width:literal) => {
        impl sealed::Sealed for $type {}

        impl Element for $type {
            const DTYPE: u8 = $dtype;

            fn encode(values: &[Self]) -> Vec<u8> {
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            }

            fn decode(bytes: &[u8]) -> Result<Vec<Self>> {
                let (values, remainder) = bytes.as_chunks::<$width>();
                if !remainder.is_empty() {
                    return Err(Error::new("host returned a partial tensor element"));
                }
                Ok(values
                    .iter()
                    .map(|bytes| <$type>::from_le_bytes(*bytes))
                    .collect())
            }
        }
    };
}

element!(f32, 0, 4);
element!(f16, 1, 2);
element!(bf16, 2, 2);
element!(u32, 3, 4);
element!(i32, 4, 4);
