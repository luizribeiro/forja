//! Component exercising tensor allocation, views, writes, and asynchronous reads.

wit_bindgen::generate!({
    path: "../../../wit",
    world: "tensor-smoke",
});

use l9o::gpu::compute::{Dtype, Tensor, ViewOp};

struct Component;

impl Guest for Component {
    async fn run() -> Result<u64, String> {
        let tensor = Tensor::alloc(Dtype::F32, &[7, 1024]).map_err(error)?;
        let mut bytes = Vec::with_capacity(7 * 1024 * 4);
        for index in 0_u16..7 * 1024 {
            bytes.extend_from_slice(&(f32::from(index) / 251.0).to_le_bytes());
        }
        tensor.write(&bytes).map_err(error)?;
        let view = tensor.view(&ViewOp::Permute(vec![1, 0])).map_err(error)?;
        let gathered = view.read().await.map_err(error)?;
        Ok(checksum(&gathered))
    }
}

fn error(error: l9o::gpu::compute::Error) -> String {
    format!("{error:?}")
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

export!(Component);
