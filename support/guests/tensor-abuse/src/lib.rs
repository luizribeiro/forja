//! Component probing refusals and resource lifetime checks at the host boundary.

wit_bindgen::generate!({
    path: "../../../wit",
    world: "tensor-abuse",
});

use l9o::gpu::compute::{Dtype, Error, SliceSpec, Tensor, ViewOp};

struct Component;

impl Guest for Component {
    async fn run() -> Result<u64, String> {
        exhaust_allocation_quota()?;

        let scalar = Tensor::alloc(Dtype::F32, &[1]).map_err(error)?;
        expect_quota(scalar.view(&ViewOp::Broadcast(vec![4_000_000_000])))?;

        let tensor = Tensor::alloc(Dtype::F32, &[7, 1024]).map_err(error)?;
        expect_layout(tensor.view(&ViewOp::Slice(vec![
            SliceSpec {
                start: 0,
                len: 7,
                step: 1,
            },
            SliceSpec {
                start: 1024,
                len: 1,
                step: 1,
            },
        ])))?;
        expect_layout(tensor.view(&ViewOp::Broadcast(vec![7, 2048])))?;
        expect_layout(tensor.write(&[]))?;

        let bytes = pattern();
        tensor.write(&bytes).map_err(error)?;
        let view = tensor.view(&ViewOp::Permute(vec![1, 0])).map_err(error)?;
        drop(tensor);
        let gathered = view.read().await.map_err(error)?;
        Ok(checksum(&gathered))
    }

    async fn large_dispatch_input() -> Result<(), String> {
        let tensor = Tensor::alloc(Dtype::F32, &[1]).map_err(error)?;
        let view = tensor
            .view(&ViewOp::Broadcast(vec![4_000_000_000]))
            .map_err(error)?;
        drop(view);
        Ok(())
    }

    async fn large_read_refused() -> Result<(), String> {
        let tensor = Tensor::alloc(Dtype::F32, &[1]).map_err(error)?;
        let view = tensor.view(&ViewOp::Broadcast(vec![4097])).map_err(error)?;
        match view.read().await {
            Err(Error::Quota(_)) => Ok(()),
            Err(error) => Err(format!("large read returned {error:?}")),
            Ok(_) => Err("large read was accepted".to_owned()),
        }
    }

    async fn grow_memory(bytes: u32) -> bool {
        let Ok(bytes) = usize::try_from(bytes) else {
            return true;
        };
        Vec::<u8>::new().try_reserve_exact(bytes).is_err()
    }

    async fn misuse_handle() {
        let Ok(tensor) = Tensor::alloc(Dtype::F32, &[1]) else {
            return;
        };
        let _ = tensor.take_handle();
        let _ = tensor.write(&[0; 4]);
    }
}

fn exhaust_allocation_quota() -> Result<(), String> {
    let mut tensors = Vec::new();
    loop {
        match Tensor::alloc(Dtype::F32, &[7, 1024]) {
            Ok(tensor) => tensors.push(tensor),
            Err(Error::Quota(_)) => break,
            Err(error) => return Err(format!("allocation returned {error:?}")),
        }
    }
    if tensors.is_empty() {
        return Err("allocation quota refused the first tensor".to_owned());
    }
    drop(tensors);
    Ok(())
}

fn expect_layout<T>(result: Result<T, Error>) -> Result<(), String> {
    match result {
        Err(Error::Layout(_)) => Ok(()),
        Err(error) => Err(format!("invalid layout returned {error:?}")),
        Ok(_) => Err("invalid layout was accepted".to_owned()),
    }
}

fn expect_quota<T>(result: Result<T, Error>) -> Result<(), String> {
    match result {
        Err(Error::Quota(_)) => Ok(()),
        Err(error) => Err(format!("over-limit tensor returned {error:?}")),
        Ok(_) => Err("over-limit tensor was accepted".to_owned()),
    }
}

fn pattern() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(7 * 1024 * 4);
    for index in 0_u16..7 * 1024 {
        bytes.extend_from_slice(&(f32::from(index) / 251.0).to_le_bytes());
    }
    bytes
}

fn error(error: Error) -> String {
    format!("{error:?}")
}

fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

export!(Component);
