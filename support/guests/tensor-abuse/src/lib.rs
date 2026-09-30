//! Component probing refusals and resource lifetime checks at the host boundary.

#![allow(
    clippy::needless_pass_by_value,
    clippy::same_length_and_capacity,
    clippy::unused_async_trait_impl
)]

wit_bindgen::generate!({
    path: "../../../wit",
    world: "tensor-abuse",
});

use l9o::gpu::compute::{
    Affine, Binop, CommandList, Dtype, Error, Graph, Inst, Kernel, KernelSignature, Op, ParamRange,
    ParamSlice, Params, ProgramKind, ProgramSource, SliceSpec, Tensor, ViewOp, replay,
};

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

    async fn churn_kernels() -> Result<u32, String> {
        let mut kernels = Vec::new();
        for value in 0_u16..100 {
            let source = ProgramSource {
                kind: ProgramKind::Map,
                insts: vec![
                    Inst::Const(f32::from(value)),
                    Inst::Const(1.0),
                    Inst::Binary((Binop::Add, 0, 1)),
                ],
                outputs: vec![(0, 2)],
            };
            let signature = KernelSignature {
                rank: 1,
                inputs: vec![],
                outputs: vec![Dtype::F32],
                scalars: 0,
            };
            match Kernel::create(&source, &signature) {
                Ok(kernel) => kernels.push(kernel),
                Err(Error::Quota(_)) => {
                    return u32::try_from(kernels.len()).map_err(|_| {
                        "kernel count exceeded the component result range".to_owned()
                    });
                }
                Err(error) => return Err(format!("kernel creation returned {error:?}")),
            }
        }
        Err("kernel churn did not reach a quota".to_owned())
    }

    async fn replay_churn() -> Result<u32, String> {
        let graph = Graph::create(CommandList::new()).map_err(error)?;
        let mut completed = 0_u32;
        loop {
            match replay(&graph, Vec::new()).await {
                Ok(_) => completed = completed.saturating_add(1),
                Err(Error::Quota(_)) => return Ok(completed),
                Err(error) => return Err(format!("graph replay returned {error:?}")),
            }
        }
    }

    async fn exhaust_graphs() -> Result<u32, String> {
        let mut graphs = Vec::new();
        loop {
            match Graph::create(CommandList::new()) {
                Ok(graph) => graphs.push(graph),
                Err(Error::Quota(_)) => {
                    return u32::try_from(graphs.len())
                        .map_err(|_| "graph count exceeded the component result range".to_owned());
                }
                Err(error) => return Err(format!("graph creation returned {error:?}")),
            }
        }
    }

    async fn rewrite_graph_buffer() -> Result<Vec<u8>, String> {
        let input = Tensor::alloc(Dtype::F32, &[1]).map_err(error)?;
        let output = Tensor::alloc(Dtype::F32, &[1]).map_err(error)?;
        let commands = CommandList::new();
        commands
            .dispatch(Op::Copy, &[&input], &output)
            .map_err(error)?;
        let graph = Graph::create(commands).map_err(error)?;
        input.write(&1.0_f32.to_le_bytes()).map_err(error)?;
        replay(&graph, Vec::new()).await.map_err(error)?;
        input.write(&2.0_f32.to_le_bytes()).map_err(error)?;
        replay(&graph, Vec::new()).await.map_err(error)?;
        output.read().await.map_err(error)
    }

    async fn retained_graph_replay() -> Result<(), String> {
        let params = Params::new(&[ParamRange { lo: 0, hi: 3 }]);
        let input = Tensor::alloc(Dtype::F32, &[4]).map_err(error)?;
        let output = Tensor::alloc(Dtype::F32, &[4]).map_err(error)?;
        input.write(&[0; 16]).map_err(error)?;
        let slice = ParamSlice {
            start: Affine {
                param: None,
                scale: 0,
                offset: 0,
            },
            len: Affine {
                param: Some(0),
                scale: 1,
                offset: 1,
            },
            step: 1,
        };
        let input_view = input.view_param(&params, &[slice]).map_err(error)?;
        let output_view = output.view_param(&params, &[slice]).map_err(error)?;
        let commands = CommandList::new();
        commands
            .dispatch(Op::Copy, &[&input_view], &output_view)
            .map_err(error)?;
        let graph = Graph::create(commands).map_err(error)?;
        drop(input_view);
        drop(output_view);
        drop(input);
        drop(output);
        drop(params);
        replay(&graph, vec![2]).await.map_err(error)?;
        Ok(())
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
