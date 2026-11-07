//! Admission coverage for real prefill shapes without allocating model weights.
use forja_config::{Limits, Unbounded};
use forja_core::{
    AllocationRegistry, CommandList, DType, GraphError, GraphLimits, GraphTemplate, Layout, Op,
    ParamSpace, Tensor,
};

fn finite(value: Unbounded<u64>) -> u64 {
    match value {
        Unbounded::Limited(n) => n,
        Unbounded::Unlimited => panic!("default work must be finite"),
    }
}

fn limits() -> GraphLimits {
    let config = Limits::default();
    let Unbounded::Limited(dispatches) = config.dispatches_per_list else {
        panic!("dispatch limit must be finite")
    };
    GraphLimits::new(
        usize::try_from(dispatches).unwrap(),
        config.tensor_elements,
        finite(config.work_per_dispatch),
    )
    .with_submission_work_limit(finite(config.work_per_submission))
}

fn tensor(registry: &mut AllocationRegistry<()>, dtype: DType, shape: &[u32]) -> Tensor {
    let bytes = shape
        .iter()
        .try_fold(dtype.byte_size(), |n, &d| n.checked_mul(u64::from(d)))
        .unwrap();
    let buffer = registry.insert((), bytes).unwrap();
    registry
        .tensor(
            buffer,
            Layout::contiguous(dtype, 0, shape.to_vec(), bytes).unwrap(),
        )
        .unwrap()
}

fn product(
    registry: &mut AllocationRegistry<()>,
    input: &Tensor,
    columns: u32,
) -> (Tensor, CommandList) {
    let weights = tensor(
        registry,
        input.layout().dtype(),
        &[input.layout().shape()[1], columns],
    );
    let output = tensor(
        registry,
        input.layout().dtype(),
        &[input.layout().shape()[0], columns],
    );
    let mut commands = CommandList::new();
    commands
        .dispatch(Op::Matmul, &[input, &weights], &output)
        .unwrap();
    (output, commands)
}

fn record(graph: &mut GraphTemplate, commands: CommandList) {
    for dispatch in commands.into_dispatches() {
        graph.record_validated(dispatch).unwrap();
    }
}

fn project(
    registry: &mut AllocationRegistry<()>,
    graph: &mut GraphTemplate,
    input: &Tensor,
    columns: u32,
) -> Tensor {
    let (output, commands) = product(registry, input, columns);
    record(graph, commands);
    output
}

fn heads(registry: &AllocationRegistry<()>, input: &Tensor, count: u32) -> Tensor {
    let layout = input
        .layout()
        .reshape(&[4096, count, 128])
        .unwrap()
        .permute(&[1, 0, 2])
        .unwrap();
    registry.view(input, layout).unwrap()
}

fn normalize(
    registry: &mut AllocationRegistry<()>,
    graph: &mut GraphTemplate,
    input: &Tensor,
) -> Tensor {
    let weights = tensor(registry, DType::BF16, &[1024]);
    let output = tensor(registry, DType::BF16, &[4096, 1024]);
    let mut commands = CommandList::new();
    commands
        .dispatch(Op::RmsNorm { eps: 1e-6 }, &[input, &weights], &output)
        .unwrap();
    record(graph, commands);
    output
}

#[test]
fn default_limits_admit_full_prefill_and_replay() {
    use forja_core::program::{
        BinOp, EpilogueSignature, Inst, Program, ProgramKind, UnOp, prepare_epilogue,
    };
    let epilogue = prepare_epilogue(
        Program {
            kind: ProgramKind::Map,
            insts: vec![
                Inst::Input(0),
                Inst::Unary(UnOp::Sigmoid, 0),
                Inst::Binary(BinOp::Mul, 0, 1),
                Inst::Input(1),
                Inst::Binary(BinOp::Mul, 2, 3),
            ],
            outputs: vec![(0, 4)],
        }
        .validate()
        .unwrap(),
        EpilogueSignature::new(1, 2, vec![DType::BF16], DType::BF16),
    )
    .unwrap();
    let mut registry = AllocationRegistry::new();
    let mut graph = GraphTemplate::new(ParamSpace::new(Vec::new()).unwrap(), limits());
    let mut hidden = tensor(&mut registry, DType::BF16, &[4096, 1024]);
    for _ in 0..28 {
        let norm = normalize(&mut registry, &mut graph, &hidden);
        let q = project(&mut registry, &mut graph, &norm, 2048);
        let k = project(&mut registry, &mut graph, &norm, 1024);
        let v = project(&mut registry, &mut graph, &norm, 1024);
        let attended = tensor(&mut registry, DType::BF16, &[4096, 2048]);
        let mut commands = CommandList::new();
        commands
            .dispatch(
                Op::Sdpa {
                    scale: 0.125,
                    causal: true,
                    q_start: 0,
                },
                &[
                    &heads(&registry, &q, 16),
                    &heads(&registry, &k, 8),
                    &heads(&registry, &v, 8),
                ],
                &heads(&registry, &attended, 16),
            )
            .unwrap();
        record(&mut graph, commands);
        let projected = project(&mut registry, &mut graph, &attended, 1024);
        let norm = normalize(&mut registry, &mut graph, &projected);
        let up = project(&mut registry, &mut graph, &norm, 3072);
        let gate = tensor(&mut registry, DType::BF16, &[1024, 3072]);
        let gated = tensor(&mut registry, DType::BF16, &[4096, 3072]);
        let mut commands = CommandList::new();
        commands
            .dispatch_epilogue(Op::Matmul, &[&norm, &gate], &gated, &epilogue, &[&up])
            .unwrap();
        record(&mut graph, commands);
        hidden = project(&mut registry, &mut graph, &gated, 1024);
    }
    project(&mut registry, &mut graph, &hidden, 151_936);
    for _ in 0..2 {
        let commands = graph
            .instantiate(&graph.values(Vec::new()).unwrap())
            .unwrap();
        limits().check_commands(&commands).unwrap();
    }
}

#[test]
fn default_limits_admit_long_context_attention_chunks() {
    let mut registry = AllocationRegistry::new();
    for length in [32_768, 262_144] {
        let query = tensor(&mut registry, DType::F32, &[64, 4096, 256]);
        let output = tensor(&mut registry, DType::F32, &[64, 4096, 256]);
        for paged in [false, true] {
            let (op, inputs) = if paged {
                let key = tensor(&mut registry, DType::F16, &[length / 64, 64, 8, 256]);
                let value = tensor(&mut registry, DType::F16, &[length / 64, 64, 8, 256]);
                let table = tensor(&mut registry, DType::U32, &[length / 64, 2]);
                let generations = tensor(&mut registry, DType::U32, &[length / 64]);
                (
                    Op::PagedSdpa {
                        scale: 0.0625,
                        causal: true,
                        q_start: length - 4096,
                        key_length: length,
                        page_tokens: 64,
                    },
                    vec![query.clone(), key, value, table, generations],
                )
            } else {
                let key = tensor(&mut registry, DType::F16, &[8, length, 256]);
                let value = tensor(&mut registry, DType::F16, &[8, length, 256]);
                (
                    Op::Sdpa {
                        scale: 0.0625,
                        causal: true,
                        q_start: length - 4096,
                    },
                    vec![query.clone(), key, value],
                )
            };
            let mut commands = CommandList::new();
            commands
                .dispatch(op, &inputs.iter().collect::<Vec<_>>(), &output)
                .unwrap();
            limits().check_commands(&commands).unwrap();
        }
    }
}

#[test]
fn default_limits_refuse_pathological_broadcast_selection() {
    let mut registry = AllocationRegistry::new();
    let scalar = tensor(&mut registry, DType::F32, &[1]);
    let input = registry
        .view(
            &scalar,
            scalar.layout().broadcast(&[1_000_000_000]).unwrap(),
        )
        .unwrap();
    let values = tensor(&mut registry, DType::F32, &[1_000_000]);
    let indices = tensor(&mut registry, DType::U32, &[1_000_000]);
    let mut commands = CommandList::new();
    commands
        .dispatch_many(
            Op::TopK {
                k: 1_000_000,
                normalize: false,
                chunk_width: 1024,
            },
            &[&input],
            &[&values, &indices],
        )
        .unwrap();
    assert_eq!(
        limits().check_commands(&commands),
        Err(GraphError::WorkLimit)
    );
}
