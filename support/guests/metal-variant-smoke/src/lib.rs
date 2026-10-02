//! Component importing the optional Metal variant interface.

#![allow(clippy::same_length_and_capacity)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "metal-variant-smoke",
});

struct Component;

impl Guest for Component {
    fn run() -> Result<(), String> {
        use l9o::gpu::{compute, metal_variants};

        let left = compute::Tensor::alloc(compute::Dtype::F32, &[1, 1])
            .map_err(|error| format!("{error:?}"))?;
        let right = compute::Tensor::alloc(compute::Dtype::F32, &[1, 1])
            .map_err(|error| format!("{error:?}"))?;
        let output = compute::Tensor::alloc(compute::Dtype::F32, &[1, 1])
            .map_err(|error| format!("{error:?}"))?;
        let commands = compute::CommandList::new();
        metal_variants::record_variant(
            &commands,
            compute::Op::Matmul,
            &[&left, &right],
            &[&output],
            "matmul.gemv",
        )
        .map_err(|error| format!("{error:?}"))
    }
}

export!(Component);
