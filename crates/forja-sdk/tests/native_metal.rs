#![cfg(all(feature = "native-metal", target_os = "macos"))]

//! Metal-backed native SDK coverage.

use forja_sdk::{NativeDevice, Slice, Tensor, set_native_device};

#[test]
fn metal_views_round_trip_non_contiguous_values() {
    set_native_device(NativeDevice::Metal);
    let tensor = Tensor::from_slice(&(0_u32..21).collect::<Vec<_>>(), &[3, 7]).unwrap();
    let view = tensor
        .slice(&[Slice::new(0, 3, 1).unwrap(), Slice::new(1, 3, 2).unwrap()])
        .unwrap()
        .t()
        .unwrap();

    assert_eq!(view.shape(), [3, 3]);
    assert_eq!(view.to_vec().unwrap(), [1, 8, 15, 3, 10, 17, 5, 12, 19]);
}
