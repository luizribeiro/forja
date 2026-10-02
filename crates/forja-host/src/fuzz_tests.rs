use std::{
    fs,
    sync::atomic::{AtomicU64, Ordering},
    time::Duration,
};

use forja_core::{
    Backend, CommandList, DType, Submission, Tensor, ViewOp,
    program::{
        BinOp, Inst, KernelSignature, Program, ProgramKind, RedOp, UnOp, ValueType, prepare_program,
    },
};
use forja_testing::{
    DeterministicValues, TensorSpec, assert_outputs_agree, generated_tensor_bytes,
    program::case_from_seed,
};
use wasmtime::component::Resource;

use super::{
    CommandListEntry, Grants, GraphEntry, Host, KernelEntry, Limits, TensorEntry,
    bindings::l9o::gpu::compute, core_dtype, guest_error,
};
use compute::{Binop as WitBinOp, Redop as WitRedOp, Unop as WitUnOp, ValueType as WitValueType};

static NEXT_PROGRAM_WEIGHT: AtomicU64 = AtomicU64::new(0);

const FUZZ_LIMITS: Limits = Limits::new(8 * 1024 * 1024, 8, 1_000_000, 128, 8 * 1024 * 1024)
    .with_command_limits(16, 1_000_000_000)
    .with_gpu_limits(Duration::from_secs(60), Duration::MAX);

#[tokio::test(flavor = "multi_thread")]
async fn symbolic_graphs_match_fresh_cpu_lists() {
    let mut coverage = GraphCoverage::default();
    for seed in 0..64 {
        coverage.record(symbolic_graph_case(forja_cpu::CpuBackend::new(), seed).await);
    }
    coverage.assert_sufficient(64);
}

#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
async fn metal_symbolic_graphs_match_fresh_lists() {
    let mut coverage = GraphCoverage::default();
    for seed in 0..32 {
        coverage.record(symbolic_graph_case(forja_metal::MetalBackend::new().unwrap(), seed).await);
    }
    coverage.assert_sufficient(32);
}

#[derive(Clone, Copy, Debug)]
enum GraphScenario {
    Elementwise,
    Matmul,
    Program,
    Sdpa,
}

#[derive(Clone, Copy, Debug)]
struct GraphCaseCoverage {
    scenario: GraphScenario,
    dtype: compute::Dtype,
    rank: usize,
    dispatches: usize,
    barriers: usize,
    replays: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct GraphCoverage {
    scenarios: [usize; 4],
    dtypes: [usize; 3],
    ranks: [usize; 3],
    dispatches: usize,
    barriers: usize,
    replays: usize,
}

impl GraphCoverage {
    fn record(&mut self, case: GraphCaseCoverage) {
        let scenario = match case.scenario {
            GraphScenario::Elementwise => 0,
            GraphScenario::Matmul => 1,
            GraphScenario::Program => 2,
            GraphScenario::Sdpa => 3,
        };
        let dtype = match case.dtype {
            compute::Dtype::F32 => 0,
            compute::Dtype::F16 => 1,
            compute::Dtype::Bf16 => 2,
            compute::Dtype::I32 | compute::Dtype::U32 => {
                panic!("graph fuzz generated a non-floating dtype")
            }
        };
        self.scenarios[scenario] += 1;
        self.dtypes[dtype] += 1;
        self.ranks[case.rank - 1] += 1;
        self.dispatches += case.dispatches;
        self.barriers += case.barriers;
        self.replays += case.replays;
    }

    fn assert_sufficient(self, cases: usize) {
        println!(
            "graph fuzz coverage: scenarios {:?}, dtypes {:?}, ranks {:?}, {} dispatches, {} barriers, {} replays",
            self.scenarios, self.dtypes, self.ranks, self.dispatches, self.barriers, self.replays,
        );
        assert!(self.scenarios.into_iter().all(|count| count > 0));
        assert!(self.dtypes.into_iter().all(|count| count > 0));
        assert!(self.ranks.into_iter().all(|count| count > 0));
        assert!(self.dispatches >= cases * 2);
        assert!(self.barriers >= cases);
        assert_eq!(self.replays, cases * 3);
    }
}

async fn symbolic_graph_case<B>(backend: B, seed: u64) -> GraphCaseCoverage
where
    B: Backend + Send + Sync + 'static,
{
    let mut values = Values::new(seed ^ 0xa076_1d64_78bd_642f);
    let dtype = [
        compute::Dtype::F32,
        compute::Dtype::F16,
        compute::Dtype::Bf16,
    ][usize::try_from(seed % 3).unwrap()];
    match seed % 4 {
        0 => graph_elementwise_case(backend, dtype, seed, &mut values).await,
        1 => graph_matmul_case(backend, dtype, seed, &mut values).await,
        2 => graph_program_case(backend, dtype, seed, &mut values).await,
        _ => graph_sdpa_case(backend, dtype, seed, &mut values).await,
    }
}

async fn graph_elementwise_case<B>(
    backend: B,
    dtype: compute::Dtype,
    seed: u64,
    values: &mut Values,
) -> GraphCaseCoverage
where
    B: Backend + Send + Sync + 'static,
{
    let rank = 1 + usize::try_from((seed / 4) % 2).unwrap();
    let shape = if rank == 1 { vec![17] } else { vec![7, 17] };
    let axis = rank - 1;
    let mut host = Host::new(backend, FUZZ_LIMITS);
    let left = initialized_tensor(&mut host, dtype, &shape, seed);
    let right = initialized_tensor(&mut host, dtype, &shape, seed.wrapping_add(1));
    let scratch = host.alloc(dtype, &shape).unwrap();
    let graph_output = host.alloc(dtype, &shape).unwrap();
    let fresh_scratch = host.alloc(dtype, &shape).unwrap();
    let fresh_output = host.alloc(dtype, &shape).unwrap();
    let params = host
        .params(vec![compute::ParamRange { lo: 0, hi: 10 }])
        .unwrap();
    let symbolic = parameter_slices(&shape, axis, affine_param(0, 1, 0), affine_const(7));
    let left_view = host.view_param(&left, &params, symbolic.clone()).unwrap();
    let right_view = host.view_param(&right, &params, symbolic.clone()).unwrap();
    let scratch_view = host
        .view_param(&scratch, &params, symbolic.clone())
        .unwrap();
    let output_view = host.view_param(&graph_output, &params, symbolic).unwrap();
    let commands = host.command_list().unwrap();
    host.dispatch(&commands, compute::Op::Copy, &[left_view], &scratch_view)
        .unwrap();
    host.dispatch(
        &commands,
        compute::Op::Add,
        &[scratch_view, right_view],
        &output_view,
    )
    .unwrap();
    let graph = host.create_graph(commands).unwrap();
    let barriers = graph_barriers(&host, &graph);
    for _ in 0..3 {
        let value = u32::try_from(values.index(11)).unwrap();
        host.prepare_replay(&graph, vec![value])
            .unwrap()
            .run()
            .await
            .unwrap();
        let concrete = concrete_slices(&shape, axis, value, 7);
        let left_view = host.view(&left, concrete.clone()).unwrap();
        let right_view = host.view(&right, concrete.clone()).unwrap();
        let scratch_view = host.view(&fresh_scratch, concrete.clone()).unwrap();
        let output_view = host.view(&fresh_output, concrete).unwrap();
        let fresh = host.command_list().unwrap();
        host.dispatch(&fresh, compute::Op::Copy, &[left_view], &scratch_view)
            .unwrap();
        host.dispatch(
            &fresh,
            compute::Op::Add,
            &[scratch_view, right_view],
            &output_view,
        )
        .unwrap();
        host.prepare_submit(fresh).unwrap().run().await.unwrap();
        compare_host_tensors(&host, dtype, &graph_output, &fresh_output).await;
    }
    GraphCaseCoverage {
        scenario: GraphScenario::Elementwise,
        dtype,
        rank,
        dispatches: 2,
        barriers,
        replays: 3,
    }
}

async fn graph_matmul_case<B>(
    backend: B,
    dtype: compute::Dtype,
    seed: u64,
    values: &mut Values,
) -> GraphCaseCoverage
where
    B: Backend + Send + Sync + 'static,
{
    let rank = 2 + usize::try_from((seed / 4) % 2).unwrap();
    let (left_shape, right_shape, output_shape, axis) = if rank == 2 {
        (vec![4, 7], vec![7, 5], vec![4, 5], 0)
    } else {
        (vec![2, 4, 7], vec![2, 7, 5], vec![2, 4, 5], 1)
    };
    let mut host = Host::new(backend, FUZZ_LIMITS);
    let left = initialized_tensor(&mut host, dtype, &left_shape, seed);
    let right = initialized_tensor(&mut host, dtype, &right_shape, seed.wrapping_add(1));
    let scratch = host.alloc(dtype, &output_shape).unwrap();
    let graph_output = host.alloc(dtype, &output_shape).unwrap();
    let fresh_scratch = host.alloc(dtype, &output_shape).unwrap();
    let fresh_output = host.alloc(dtype, &output_shape).unwrap();
    let params = host
        .params(vec![compute::ParamRange { lo: 0, hi: 3 }])
        .unwrap();
    let left_symbolic = parameter_slices(&left_shape, axis, affine_const(0), affine_param(0, 1, 1));
    let output_symbolic =
        parameter_slices(&output_shape, axis, affine_const(0), affine_param(0, 1, 1));
    let left_view = host.view_param(&left, &params, left_symbolic).unwrap();
    let scratch_view = host
        .view_param(&scratch, &params, output_symbolic.clone())
        .unwrap();
    let output_view = host
        .view_param(&graph_output, &params, output_symbolic)
        .unwrap();
    let commands = host.command_list().unwrap();
    host.dispatch(
        &commands,
        compute::Op::Matmul,
        &[left_view, Resource::new_borrow(right.rep())],
        &scratch_view,
    )
    .unwrap();
    host.dispatch(&commands, compute::Op::Copy, &[scratch_view], &output_view)
        .unwrap();
    let graph = host.create_graph(commands).unwrap();
    let barriers = graph_barriers(&host, &graph);
    for _ in 0..3 {
        let value = u32::try_from(values.index(4)).unwrap();
        host.prepare_replay(&graph, vec![value])
            .unwrap()
            .run()
            .await
            .unwrap();
        let extent = value + 1;
        let left_view = host
            .view(&left, concrete_slices(&left_shape, axis, 0, extent))
            .unwrap();
        let scratch_view = host
            .view(
                &fresh_scratch,
                concrete_slices(&output_shape, axis, 0, extent),
            )
            .unwrap();
        let output_view = host
            .view(
                &fresh_output,
                concrete_slices(&output_shape, axis, 0, extent),
            )
            .unwrap();
        let fresh = host.command_list().unwrap();
        host.dispatch(
            &fresh,
            compute::Op::Matmul,
            &[left_view, Resource::new_borrow(right.rep())],
            &scratch_view,
        )
        .unwrap();
        host.dispatch(&fresh, compute::Op::Copy, &[scratch_view], &output_view)
            .unwrap();
        host.prepare_submit(fresh).unwrap().run().await.unwrap();
        compare_host_tensors(&host, dtype, &graph_output, &fresh_output).await;
    }
    GraphCaseCoverage {
        scenario: GraphScenario::Matmul,
        dtype,
        rank,
        dispatches: 2,
        barriers,
        replays: 3,
    }
}

async fn graph_program_case<B>(
    backend: B,
    dtype: compute::Dtype,
    seed: u64,
    values: &mut Values,
) -> GraphCaseCoverage
where
    B: Backend + Send + Sync + 'static,
{
    let rank = 1 + usize::try_from((seed / 4) % 2).unwrap();
    let shape = if rank == 1 { vec![17] } else { vec![7, 17] };
    let axis = rank - 1;
    let mut host = Host::new(backend, FUZZ_LIMITS);
    let input = initialized_tensor(&mut host, dtype, &shape, seed);
    let scratch = host.alloc(dtype, &shape).unwrap();
    let graph_output = host.alloc(dtype, &shape).unwrap();
    let fresh_scratch = host.alloc(dtype, &shape).unwrap();
    let fresh_output = host.alloc(dtype, &shape).unwrap();
    let kernel = host
        .create_kernel(
            compute::ProgramSource {
                kind: compute::ProgramKind::Map,
                insts: vec![
                    compute::Inst::Input(0),
                    compute::Inst::Const(2.0),
                    compute::Inst::Binary((compute::Binop::Mul, 0, 1)),
                ],
                outputs: vec![(0, 2)],
            },
            compute::KernelSignature {
                rank: u8::try_from(rank).unwrap(),
                inputs: vec![dtype],
                outputs: vec![dtype],
                scalars: 0,
            },
        )
        .unwrap();
    let params = host
        .params(vec![compute::ParamRange { lo: 0, hi: 10 }])
        .unwrap();
    let symbolic = parameter_slices(&shape, axis, affine_param(0, 1, 0), affine_const(7));
    let input_view = host.view_param(&input, &params, symbolic.clone()).unwrap();
    let scratch_view = host
        .view_param(&scratch, &params, symbolic.clone())
        .unwrap();
    let output_view = host.view_param(&graph_output, &params, symbolic).unwrap();
    let commands = host.command_list().unwrap();
    host.dispatch_kernel(
        &commands,
        &kernel,
        &[input_view],
        &[Resource::new_borrow(scratch_view.rep())],
    )
    .unwrap();
    host.dispatch(&commands, compute::Op::Copy, &[scratch_view], &output_view)
        .unwrap();
    let graph = host.create_graph(commands).unwrap();
    let barriers = graph_barriers(&host, &graph);
    for _ in 0..3 {
        let value = u32::try_from(values.index(11)).unwrap();
        host.prepare_replay(&graph, vec![value])
            .unwrap()
            .run()
            .await
            .unwrap();
        let concrete = concrete_slices(&shape, axis, value, 7);
        let input_view = host.view(&input, concrete.clone()).unwrap();
        let scratch_view = host.view(&fresh_scratch, concrete.clone()).unwrap();
        let output_view = host.view(&fresh_output, concrete).unwrap();
        let fresh = host.command_list().unwrap();
        host.dispatch_kernel(
            &fresh,
            &kernel,
            &[input_view],
            &[Resource::new_borrow(scratch_view.rep())],
        )
        .unwrap();
        host.dispatch(&fresh, compute::Op::Copy, &[scratch_view], &output_view)
            .unwrap();
        host.prepare_submit(fresh).unwrap().run().await.unwrap();
        compare_host_tensors(&host, dtype, &graph_output, &fresh_output).await;
    }
    GraphCaseCoverage {
        scenario: GraphScenario::Program,
        dtype,
        rank,
        dispatches: 2,
        barriers,
        replays: 3,
    }
}

async fn graph_sdpa_case<B>(
    backend: B,
    dtype: compute::Dtype,
    seed: u64,
    values: &mut Values,
) -> GraphCaseCoverage
where
    B: Backend + Send + Sync + 'static,
{
    let mut host = Host::new(backend, FUZZ_LIMITS);
    let query = initialized_tensor(&mut host, dtype, &[2, 1, 128], seed);
    let source_key = initialized_tensor(&mut host, dtype, &[1, 1, 128], seed.wrapping_add(1));
    let key_cache = initialized_tensor(&mut host, dtype, &[1, 7, 128], seed.wrapping_add(2));
    let value_cache = initialized_tensor(&mut host, dtype, &[1, 7, 128], seed.wrapping_add(3));
    let graph_output = host.alloc(dtype, &[2, 1, 128]).unwrap();
    let fresh_key_cache = initialized_tensor(&mut host, dtype, &[1, 7, 128], seed.wrapping_add(2));
    let fresh_output = host.alloc(dtype, &[2, 1, 128]).unwrap();
    let params = host
        .params(vec![compute::ParamRange { lo: 0, hi: 6 }])
        .unwrap();
    let key_slot = parameter_slices(&[1, 7, 128], 1, affine_param(0, 1, 0), affine_const(1));
    let prefix = parameter_slices(&[1, 7, 128], 1, affine_const(0), affine_param(0, 1, 1));
    let key_slot = host.view_param(&key_cache, &params, key_slot).unwrap();
    let key_prefix = host
        .view_param(&key_cache, &params, prefix.clone())
        .unwrap();
    let value_prefix = host.view_param(&value_cache, &params, prefix).unwrap();
    let commands = host.command_list().unwrap();
    host.dispatch(
        &commands,
        compute::Op::Copy,
        &[Resource::new_borrow(source_key.rep())],
        &key_slot,
    )
    .unwrap();
    host.dispatch(
        &commands,
        compute::Op::Sdpa(compute::SdpaCfg {
            scale: 0.088_388_346,
            causal: true,
            q_start: affine_param(0, 1, 0),
        }),
        &[Resource::new_borrow(query.rep()), key_prefix, value_prefix],
        &graph_output,
    )
    .unwrap();
    let graph = host.create_graph(commands).unwrap();
    let barriers = graph_barriers(&host, &graph);
    for _ in 0..3 {
        let value = u32::try_from(values.index(7)).unwrap();
        host.prepare_replay(&graph, vec![value])
            .unwrap()
            .run()
            .await
            .unwrap();
        let key_slot = host
            .view(&fresh_key_cache, concrete_slices(&[1, 7, 128], 1, value, 1))
            .unwrap();
        let key_prefix = host
            .view(
                &fresh_key_cache,
                concrete_slices(&[1, 7, 128], 1, 0, value + 1),
            )
            .unwrap();
        let value_prefix = host
            .view(&value_cache, concrete_slices(&[1, 7, 128], 1, 0, value + 1))
            .unwrap();
        let fresh = host.command_list().unwrap();
        host.dispatch(
            &fresh,
            compute::Op::Copy,
            &[Resource::new_borrow(source_key.rep())],
            &key_slot,
        )
        .unwrap();
        host.dispatch(
            &fresh,
            compute::Op::Sdpa(compute::SdpaCfg {
                scale: 0.088_388_346,
                causal: true,
                q_start: affine_const(value),
            }),
            &[Resource::new_borrow(query.rep()), key_prefix, value_prefix],
            &fresh_output,
        )
        .unwrap();
        host.prepare_submit(fresh).unwrap().run().await.unwrap();
        compare_host_tensors(&host, dtype, &graph_output, &fresh_output).await;
    }
    GraphCaseCoverage {
        scenario: GraphScenario::Sdpa,
        dtype,
        rank: 3,
        dispatches: 2,
        barriers,
        replays: 3,
    }
}

fn initialized_tensor<B: Backend + 'static>(
    host: &mut Host<B>,
    dtype: compute::Dtype,
    shape: &[u32],
    seed: u64,
) -> Resource<TensorEntry> {
    let tensor = host.alloc(dtype, shape).unwrap();
    let spec = TensorSpec::contiguous(core_dtype(dtype), shape);
    let bytes = generated_tensor_bytes(&spec, &mut DeterministicValues::new(seed)).unwrap();
    host.write(&tensor, &bytes).unwrap();
    tensor
}

const fn affine_const(offset: u32) -> compute::Affine {
    compute::Affine {
        param: None,
        scale: 0,
        offset,
    }
}

const fn affine_param(param: u8, scale: u32, offset: u32) -> compute::Affine {
    compute::Affine {
        param: Some(param),
        scale,
        offset,
    }
}

fn parameter_slices(
    shape: &[u32],
    axis: usize,
    start: compute::Affine,
    len: compute::Affine,
) -> Vec<compute::ParamSlice> {
    shape
        .iter()
        .enumerate()
        .map(|(index, &extent)| compute::ParamSlice {
            start: if index == axis {
                start
            } else {
                affine_const(0)
            },
            len: if index == axis {
                len
            } else {
                affine_const(extent)
            },
            step: 1,
        })
        .collect()
}

fn concrete_slices(shape: &[u32], axis: usize, start: u32, len: u32) -> compute::ViewOp {
    compute::ViewOp::Slice(
        shape
            .iter()
            .enumerate()
            .map(|(index, &extent)| compute::SliceSpec {
                start: if index == axis { start } else { 0 },
                len: if index == axis { len } else { extent },
                step: 1,
            })
            .collect(),
    )
}

fn graph_barriers<B: Backend>(host: &Host<B>, graph: &Resource<GraphEntry>) -> usize {
    host.table
        .get(graph)
        .unwrap()
        .graph
        .template()
        .required_barriers()
        .iter()
        .filter(|&&barrier| barrier)
        .count()
}

async fn compare_host_tensors<B>(
    host: &Host<B>,
    dtype: compute::Dtype,
    actual: &Resource<TensorEntry>,
    expected: &Resource<TensorEntry>,
) where
    B: Backend + Send + Sync + 'static,
{
    let actual = host.prepare_read(actual).unwrap().run().await.unwrap();
    let expected = host.prepare_read(expected).unwrap().run().await.unwrap();
    assert_outputs_agree(core_dtype(dtype), &expected, &actual).unwrap();
}

#[derive(Clone, Debug)]
enum Action {
    Alloc(compute::Dtype, Vec<u32>),
    View(usize, compute::ViewOp),
    Write(usize, Vec<u8>),
    NewList,
    Dispatch {
        list: usize,
        operation: compute::Op,
        inputs: Vec<usize>,
        output: usize,
    },
    Submit(usize),
    Read(usize),
    DropTensor(usize),
    DropList(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ErrorKind {
    Layout,
    OpSignature,
    Quota,
    BackendExecution,
    InvalidHandle,
}

#[derive(Debug, Eq, PartialEq)]
enum Outcome {
    Unit(Result<(), ErrorKind>),
    Submit(Result<(), ErrorKind>),
    Read(Result<ReadValue, ErrorKind>),
}

#[derive(Debug, Eq, PartialEq)]
struct ReadValue {
    bytes: Vec<u8>,
    dtype: compute::Dtype,
    operation_output: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct Coverage {
    dispatch_attempts: usize,
    accepted_dispatches: usize,
    successful_submits: usize,
    compared_reads: usize,
}

impl Coverage {
    fn assert_sufficient(self, cases: usize) {
        println!(
            "WIT fuzz coverage: {} accepted / {} dispatches, {} submits, {} compared reads",
            self.accepted_dispatches,
            self.dispatch_attempts,
            self.successful_submits,
            self.compared_reads,
        );
        assert!(self.dispatch_attempts > 0);
        assert!(self.accepted_dispatches * 5 >= self.dispatch_attempts);
        assert!(self.successful_submits >= cases);
        assert!(self.compared_reads * 10 >= cases);
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ProgramCoverage {
    accepted_dispatches: usize,
    aliasing_refusals: usize,
    read_only_refusals: usize,
    signature_refusals: usize,
    compared_reads: usize,
}

impl ProgramCoverage {
    fn assert_sufficient(self, cases: usize) {
        println!(
            "WIT program fuzz coverage: {} accepted, {} aliasing refusals, {} read-only refusals, {} signature refusals, {} compared reads",
            self.accepted_dispatches,
            self.aliasing_refusals,
            self.read_only_refusals,
            self.signature_refusals,
            self.compared_reads,
        );
        assert_eq!(
            self.accepted_dispatches + self.aliasing_refusals + self.read_only_refusals,
            cases
        );
        assert!(self.accepted_dispatches * 4 >= cases * 3);
        assert!(self.aliasing_refusals * 8 >= cases);
        assert!(self.read_only_refusals * 8 >= cases);
        assert_eq!(self.signature_refusals, self.accepted_dispatches);
        assert!(self.compared_reads >= self.accepted_dispatches);
    }
}

struct TensorSlot {
    resource: Option<Resource<TensorEntry>>,
    dtype: compute::Dtype,
    byte_len: usize,
    operation_output: bool,
}

struct Harness<B: Backend> {
    host: Host<B>,
    tensors: Vec<TensorSlot>,
    lists: Vec<Option<Resource<CommandListEntry>>>,
}

impl<B> Harness<B>
where
    B: Backend + Send + Sync + 'static,
{
    fn new(backend: B) -> Self {
        Self {
            host: Host::new(backend, FUZZ_LIMITS),
            tensors: Vec::new(),
            lists: Vec::new(),
        }
    }

    async fn apply(&mut self, action: &Action) -> Outcome {
        match action {
            Action::Alloc(dtype, shape) => self.alloc(*dtype, shape),
            Action::View(source, operation) => self.view(*source, operation.clone()),
            Action::Write(tensor, bytes) => Outcome::Unit(
                self.host
                    .write(&self.tensor(*tensor), bytes)
                    .map_err(|error| error_kind(&error)),
            ),
            Action::NewList => self.new_list(),
            Action::Dispatch {
                list,
                operation,
                inputs,
                output,
            } => self.dispatch(*list, *operation, inputs, *output),
            Action::Submit(list) => self.submit(*list).await,
            Action::Read(tensor) => self.read(*tensor).await,
            Action::DropTensor(tensor) => self.drop_tensor(*tensor),
            Action::DropList(list) => self.drop_list(*list),
        }
    }

    fn alloc(&mut self, dtype: compute::Dtype, shape: &[u32]) -> Outcome {
        let result = self.host.alloc(dtype, shape);
        let outcome = result.as_ref().map(|_| ()).map_err(error_kind);
        self.tensors.push(TensorSlot {
            resource: result.ok(),
            dtype,
            byte_len: logical_byte_len(dtype, shape),
            operation_output: false,
        });
        Outcome::Unit(outcome)
    }

    fn view(&mut self, source: usize, operation: compute::ViewOp) -> Outcome {
        let source_resource = self.tensor(source);
        let result = self.host.view(&source_resource, operation);
        let outcome = result.as_ref().map(|_| ()).map_err(error_kind);
        let (dtype, byte_len) = result
            .as_ref()
            .map_or((compute::Dtype::F32, 0), |resource| {
                let tensor = &self.host.entry(resource).unwrap().tensor;
                (
                    guest_dtype(tensor.layout().dtype()),
                    usize::try_from(
                        tensor
                            .layout()
                            .element_count()
                            .saturating_mul(tensor.layout().dtype().byte_size()),
                    )
                    .unwrap_or(usize::MAX),
                )
            });
        self.tensors.push(TensorSlot {
            resource: result.ok(),
            dtype,
            byte_len,
            operation_output: false,
        });
        Outcome::Unit(outcome)
    }

    fn new_list(&mut self) -> Outcome {
        let result = self.host.command_list();
        let outcome = result.as_ref().map(|_| ()).map_err(error_kind);
        self.lists.push(result.ok());
        Outcome::Unit(outcome)
    }

    fn dispatch(
        &mut self,
        list: usize,
        operation: compute::Op,
        inputs: &[usize],
        output: usize,
    ) -> Outcome {
        let input_resources = inputs
            .iter()
            .map(|&index| self.tensor(index))
            .collect::<Vec<_>>();
        let result = self.host.dispatch(
            &self.list(list),
            operation,
            &input_resources,
            &self.tensor(output),
        );
        if result.is_ok()
            && let Some(slot) = self.tensors.get_mut(output)
        {
            slot.operation_output = true;
        }
        Outcome::Unit(result.map_err(|error| error_kind(&error)))
    }

    async fn submit(&mut self, list: usize) -> Outcome {
        let resource = self
            .lists
            .get_mut(list)
            .and_then(Option::take)
            .unwrap_or_else(|| Resource::new_own(foreign_rep(list)));
        let request = self.host.prepare_submit(resource);
        let result = match request {
            Ok(request) => request.run().await.map(|_| ()),
            Err(error) => Err(error),
        };
        Outcome::Submit(result.map_err(|error| error_kind(&error)))
    }

    async fn read(&self, tensor: usize) -> Outcome {
        let resource = self.tensor(tensor);
        let request = self.host.prepare_read(&resource);
        let result = match request {
            Ok(request) => request.run().await.map_err(guest_error),
            Err(error) => Err(error),
        };
        let slot = self.tensors.get(tensor);
        Outcome::Read(
            result
                .map(|bytes| ReadValue {
                    bytes,
                    dtype: slot.map_or(compute::Dtype::F32, |slot| slot.dtype),
                    operation_output: slot.is_some_and(|slot| slot.operation_output),
                })
                .map_err(|error| error_kind(&error)),
        )
    }

    fn drop_tensor(&mut self, tensor: usize) -> Outcome {
        let resource = self
            .tensors
            .get_mut(tensor)
            .and_then(|slot| slot.resource.take())
            .unwrap_or_else(|| Resource::new_own(foreign_rep(tensor)));
        Outcome::Unit(
            self.host
                .drop_tensor(resource)
                .map_err(|error| error_kind(&error)),
        )
    }

    fn drop_list(&mut self, list: usize) -> Outcome {
        let resource = self
            .lists
            .get_mut(list)
            .and_then(Option::take)
            .unwrap_or_else(|| Resource::new_own(foreign_rep(list)));
        Outcome::Unit(
            self.host
                .drop_command_list(resource)
                .map_err(|error| error_kind(&error)),
        )
    }

    fn tensor(&self, index: usize) -> Resource<TensorEntry> {
        self.tensors
            .get(index)
            .and_then(|slot| slot.resource.as_ref())
            .map_or_else(
                || Resource::new_borrow(foreign_rep(index)),
                |resource| Resource::new_borrow(resource.rep()),
            )
    }

    fn list(&self, index: usize) -> Resource<CommandListEntry> {
        self.lists.get(index).and_then(Option::as_ref).map_or_else(
            || Resource::new_borrow(foreign_rep(index)),
            |resource| Resource::new_borrow(resource.rep()),
        )
    }
}

#[derive(Clone, Debug)]
struct Values(u64);

impl Values {
    const fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn index(&mut self, upper: usize) -> usize {
        usize::try_from(self.next() % u64::try_from(upper.max(1)).unwrap()).unwrap()
    }
}

async fn run_cases<B, F>(cases: usize, random_steps: usize, mut backend: F) -> Coverage
where
    B: Backend + Send + Sync + 'static,
    F: FnMut() -> B,
{
    let mut coverage = Coverage::default();
    for case in 0..cases {
        let mut candidate = Harness::new(backend());
        let mut reference = Harness::new(forja_cpu::CpuBackend::new());
        for action in canonical_actions() {
            replay(&mut candidate, &mut reference, &action, &mut coverage).await;
        }
        let mut values = Values::new(0x9e37_79b9_7f4a_7c15 ^ case as u64);
        for _ in 0..random_steps {
            let action = random_action(&mut values, &candidate);
            replay(&mut candidate, &mut reference, &action, &mut coverage).await;
        }
    }
    coverage
}

async fn replay<B: Backend + Send + Sync + 'static>(
    candidate: &mut Harness<B>,
    reference: &mut Harness<forja_cpu::CpuBackend>,
    action: &Action,
    coverage: &mut Coverage,
) {
    let actual = candidate.apply(action).await;
    let expected = reference.apply(action).await;
    if matches!(action, Action::Dispatch { .. }) {
        coverage.dispatch_attempts += 1;
        if matches!(&actual, Outcome::Unit(Ok(()))) {
            coverage.accepted_dispatches += 1;
        }
    }
    if matches!(action, Action::Submit(_)) && matches!(&actual, Outcome::Submit(Ok(()))) {
        coverage.successful_submits += 1;
    }
    if let Outcome::Read(Ok(actual)) = actual {
        let Outcome::Read(Ok(expected)) = expected else {
            panic!("{action:?}: candidate read succeeded but CPU returned {expected:?}");
        };
        assert_eq!(actual.dtype, expected.dtype, "{action:?}");
        assert_eq!(
            actual.operation_output, expected.operation_output,
            "{action:?}"
        );
        if actual.operation_output {
            assert_outputs_agree(core_dtype(actual.dtype), &expected.bytes, &actual.bytes)
                .unwrap_or_else(|error| panic!("{action:?}: {error}"));
            coverage.compared_reads += 1;
        } else {
            assert_eq!(actual.bytes, expected.bytes, "{action:?}");
        }
    }
}

fn canonical_actions() -> Vec<Action> {
    let specs = [
        (compute::Dtype::F32, vec![7]),
        (compute::Dtype::F32, vec![7]),
        (compute::Dtype::F16, vec![7]),
        (compute::Dtype::F16, vec![7]),
        (compute::Dtype::F16, vec![7]),
        (compute::Dtype::Bf16, vec![33]),
        (compute::Dtype::Bf16, vec![33]),
        (compute::Dtype::Bf16, vec![33]),
        (compute::Dtype::F32, vec![7, 1024]),
        (compute::Dtype::F32, vec![1024]),
        (compute::Dtype::F32, vec![7, 1024]),
        (compute::Dtype::F32, vec![7, 33]),
        (compute::Dtype::F32, vec![7, 33]),
        (compute::Dtype::F32, vec![7, 16, 128]),
        (compute::Dtype::U32, vec![7]),
        (compute::Dtype::F32, vec![7, 16, 128]),
        (compute::Dtype::F32, vec![33, 128]),
        (compute::Dtype::U32, vec![7]),
        (compute::Dtype::F32, vec![7, 128]),
        (compute::Dtype::F32, vec![7, 33]),
        (compute::Dtype::F32, vec![33, 1024]),
        (compute::Dtype::F32, vec![7, 1024]),
        (compute::Dtype::F32, vec![16, 7, 128]),
        (compute::Dtype::F32, vec![8, 33, 128]),
        (compute::Dtype::F32, vec![8, 33, 128]),
        (compute::Dtype::F32, vec![16, 7, 128]),
    ];
    let mut actions = specs
        .iter()
        .cloned()
        .map(|(dtype, shape)| Action::Alloc(dtype, shape))
        .collect::<Vec<_>>();
    for index in [0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 17, 19, 20, 22, 23, 24] {
        let (dtype, shape) = &specs[index];
        actions.push(Action::Write(
            index,
            constant_bytes(*dtype, element_count(shape)),
        ));
    }
    actions.push(Action::View(11, compute::ViewOp::Permute(vec![1, 0])));
    actions.push(Action::NewList);
    actions.extend([
        dispatch(0, compute::Op::Copy, &[0], 1),
        dispatch(0, compute::Op::Add, &[2, 3], 4),
        dispatch(0, compute::Op::SiluMul, &[5, 6], 7),
        dispatch(0, compute::Op::RmsNorm(0.00001), &[8, 9], 10),
        dispatch(0, compute::Op::Softmax, &[11], 12),
        dispatch(
            0,
            compute::Op::Rope(compute::RopeCfg { theta: 10_000.0 }),
            &[13, 14],
            15,
        ),
        dispatch(0, compute::Op::Embed, &[16, 17], 18),
        dispatch(0, compute::Op::Matmul, &[19, 20], 21),
        dispatch(
            0,
            compute::Op::Sdpa(compute::SdpaCfg {
                scale: 0.088,
                causal: false,
                q_start: constant_affine(0),
            }),
            &[22, 23, 24],
            25,
        ),
        Action::Submit(0),
    ]);
    actions.extend([1, 4, 7, 10, 12, 15, 18, 21, 25].map(Action::Read));
    actions.extend([
        Action::NewList,
        Action::Alloc(compute::Dtype::F32, vec![7]),
        Action::Write(27, constant_bytes(compute::Dtype::F32, 7)),
        Action::Alloc(compute::Dtype::F32, vec![7]),
        dispatch(1, compute::Op::Copy, &[27], 28),
        dispatch(1, compute::Op::Copy, &[27], 27),
        dispatch(1, compute::Op::Copy, &[usize::MAX], 28),
        Action::Submit(1),
        Action::Read(28),
        Action::NewList,
        Action::DropList(2),
        Action::DropTensor(26),
    ]);
    actions
}

fn dispatch(list: usize, operation: compute::Op, inputs: &[usize], output: usize) -> Action {
    Action::Dispatch {
        list,
        operation,
        inputs: inputs.to_vec(),
        output,
    }
}

fn random_action<B: Backend>(values: &mut Values, harness: &Harness<B>) -> Action {
    match values.index(9) {
        0 => {
            let dtype = random_dtype(values);
            Action::Alloc(dtype, random_shape(values))
        }
        1 => Action::View(
            random_slot(values, harness.tensors.len()),
            random_view(values),
        ),
        2 => {
            let tensor = random_slot(values, harness.tensors.len());
            let valid_len = harness.tensors.get(tensor).map_or(0, |slot| slot.byte_len);
            let len = if values.next().is_multiple_of(3) {
                valid_len
            } else {
                values.index(67)
            };
            let dtype = harness
                .tensors
                .get(tensor)
                .map_or(compute::Dtype::F32, |slot| slot.dtype);
            Action::Write(tensor, random_bytes(values, dtype, len))
        }
        3 => Action::NewList,
        4 => {
            let input_count = values.index(5);
            let mut inputs = (0..input_count)
                .map(|_| random_slot(values, harness.tensors.len()))
                .collect::<Vec<_>>();
            let output = if !inputs.is_empty() && values.next().is_multiple_of(4) {
                inputs[0]
            } else {
                random_slot(values, harness.tensors.len())
            };
            if values.next().is_multiple_of(7) {
                inputs.push(usize::MAX);
            }
            Action::Dispatch {
                list: random_slot(values, harness.lists.len()),
                operation: random_op(values),
                inputs,
                output,
            }
        }
        5 => Action::Submit(random_slot(values, harness.lists.len())),
        6 => Action::Read(random_slot(values, harness.tensors.len())),
        7 if values.next().is_multiple_of(2) => {
            Action::DropTensor(random_slot(values, harness.tensors.len()))
        }
        _ => Action::DropList(random_slot(values, harness.lists.len())),
    }
}

fn random_dtype(values: &mut Values) -> compute::Dtype {
    [
        compute::Dtype::F32,
        compute::Dtype::F16,
        compute::Dtype::Bf16,
        compute::Dtype::I32,
        compute::Dtype::U32,
    ][values.index(5)]
}

fn random_shape(values: &mut Values) -> Vec<u32> {
    let shapes = [
        vec![],
        vec![0],
        vec![1],
        vec![7],
        vec![33],
        vec![7, 33],
        vec![33, 7],
        vec![7, 16, 128],
        vec![16, 7, 128],
        vec![8, 33, 128],
    ];
    shapes[values.index(shapes.len())].clone()
}

fn random_view(values: &mut Values) -> compute::ViewOp {
    match values.index(4) {
        0 => compute::ViewOp::Slice(
            (0..values.index(5))
                .map(|_| compute::SliceSpec {
                    start: u32::try_from(values.index(9)).unwrap(),
                    len: u32::try_from(values.index(34)).unwrap(),
                    step: u32::try_from(values.index(4)).unwrap(),
                })
                .collect(),
        ),
        1 => compute::ViewOp::Reshape(random_shape(values)),
        2 => compute::ViewOp::Permute(
            (0..values.index(5))
                .map(|_| u8::try_from(values.index(5)).unwrap())
                .collect(),
        ),
        _ => compute::ViewOp::Broadcast(random_shape(values)),
    }
}

fn random_op(values: &mut Values) -> compute::Op {
    match values.index(9) {
        0 => compute::Op::Copy,
        1 => compute::Op::Add,
        2 => compute::Op::SiluMul,
        3 => compute::Op::RmsNorm(random_config(values)),
        4 => compute::Op::Softmax,
        5 => compute::Op::Rope(compute::RopeCfg {
            theta: random_config(values),
        }),
        6 => compute::Op::Embed,
        7 => compute::Op::Matmul,
        _ => compute::Op::Sdpa(compute::SdpaCfg {
            scale: random_config(values),
            causal: values.next().is_multiple_of(2),
            q_start: constant_affine(u32::try_from(values.index(40)).unwrap()),
        }),
    }
}

fn constant_affine(offset: u32) -> compute::Affine {
    compute::Affine {
        param: None,
        scale: 0,
        offset,
    }
}

fn random_config(values: &mut Values) -> f32 {
    match values.index(8) {
        0 => f32::NAN,
        1 => f32::INFINITY,
        2 => -1.0,
        _ => (f32::from(u16::try_from(values.index(1000)).unwrap()) + 1.0) / 100.0,
    }
}

fn random_slot(values: &mut Values, len: usize) -> usize {
    if values.next().is_multiple_of(8) {
        usize::MAX
    } else {
        values.index(len.saturating_add(2))
    }
}

fn random_bytes(values: &mut Values, dtype: compute::Dtype, len: usize) -> Vec<u8> {
    if dtype == compute::Dtype::U32 {
        return vec![0; len];
    }
    (0..len)
        .map(|_| u8::try_from(values.index(16)).unwrap())
        .collect()
}

fn constant_bytes(dtype: compute::Dtype, elements: usize) -> Vec<u8> {
    let scalar: &[u8] = match dtype {
        compute::Dtype::F32 => &1.0_f32.to_le_bytes(),
        compute::Dtype::F16 => &0x3c00_u16.to_le_bytes(),
        compute::Dtype::Bf16 => &0x3f80_u16.to_le_bytes(),
        compute::Dtype::I32 => &1_i32.to_le_bytes(),
        compute::Dtype::U32 => &0_u32.to_le_bytes(),
    };
    scalar.repeat(elements)
}

fn element_count(shape: &[u32]) -> usize {
    shape
        .iter()
        .try_fold(1_usize, |count, &extent| {
            count.checked_mul(usize::try_from(extent).ok()?)
        })
        .unwrap()
}

fn logical_byte_len(dtype: compute::Dtype, shape: &[u32]) -> usize {
    element_count(shape).saturating_mul(usize::try_from(core_dtype(dtype).byte_size()).unwrap())
}

const fn guest_dtype(dtype: forja_core::DType) -> compute::Dtype {
    match dtype {
        forja_core::DType::F32 => compute::Dtype::F32,
        forja_core::DType::F16 => compute::Dtype::F16,
        forja_core::DType::BF16 => compute::Dtype::Bf16,
        forja_core::DType::I32 => compute::Dtype::I32,
        forja_core::DType::U32 => compute::Dtype::U32,
    }
}

fn foreign_rep(index: usize) -> u32 {
    u32::MAX - u32::try_from(index & 0xffff).unwrap()
}

fn error_kind(error: &compute::Error) -> ErrorKind {
    match error {
        compute::Error::Layout(_) => ErrorKind::Layout,
        compute::Error::OpSignature(_) => ErrorKind::OpSignature,
        compute::Error::Quota(_) => ErrorKind::Quota,
        compute::Error::BackendExecution(_) => ErrorKind::BackendExecution,
        compute::Error::InvalidHandle(_) => ErrorKind::InvalidHandle,
    }
}

async fn run_program_case(seed: u64, coverage: &mut ProgramCoverage) {
    let case = case_from_seed(seed, 64);
    let program = case.program().clone();
    let validated = program.validate().unwrap();
    let path = program_weight_file(&case.outputs()[0]);
    let grants = Grants::new().with_weights("program-output", &path);
    let mut host = Host::with_grants(forja_cpu::CpuBackend::new(), FUZZ_LIMITS, grants);
    let reference = forja_cpu::CpuBackend::new();
    let mut values = DeterministicValues::new(seed ^ 0x6a09_e667_f3bc_c909);
    let input_bytes = case
        .inputs()
        .iter()
        .map(|spec| generated_tensor_bytes(spec, &mut values).unwrap())
        .collect::<Vec<_>>();
    let host_inputs = case
        .inputs()
        .iter()
        .zip(&input_bytes)
        .map(|(spec, bytes)| allocate_host_program_tensor(&mut host, spec, Some(bytes)))
        .collect::<Vec<_>>();
    let reference_inputs = case
        .inputs()
        .iter()
        .zip(&input_bytes)
        .map(|(spec, bytes)| allocate_reference_program_tensor(&reference, spec, Some(bytes)))
        .collect::<Vec<_>>();
    let mut host_outputs = case
        .outputs()
        .iter()
        .map(|spec| allocate_host_program_tensor(&mut host, spec, None))
        .collect::<Vec<_>>();
    let binding = seed % 8;
    if binding == 0 {
        host_outputs[0] = Resource::new_borrow(host_inputs[0].rep());
    } else if binding == 1 {
        let weights = host.open_weights("program-output").unwrap();
        host_outputs[0] = host.weight_tensor(&weights, "value").unwrap();
    }
    let kernel = host
        .create_kernel(wit_program(program.clone()), wit_signature(&case))
        .unwrap();
    let commands = host.command_list().unwrap();
    let result = host.dispatch_kernel(&commands, &kernel, &host_inputs, &host_outputs);
    if binding == 0 {
        assert!(matches!(result, Err(compute::Error::OpSignature(_))));
        coverage.aliasing_refusals += 1;
        drop(host);
        fs::remove_file(path).unwrap();
        return;
    }
    if binding == 1 {
        assert!(matches!(result, Err(compute::Error::OpSignature(_))));
        coverage.read_only_refusals += 1;
        drop(host);
        fs::remove_file(path).unwrap();
        return;
    }
    result.unwrap();
    coverage.accepted_dispatches += 1;
    let second_outputs = reuse_and_drop_kernel(
        &mut host,
        &commands,
        kernel,
        program,
        &case,
        &host_inputs,
        coverage,
    );
    host.prepare_submit(commands).unwrap().run().await.unwrap();

    let (reference_outputs, second_reference_outputs) =
        run_reference_program(&reference, &case, validated, &reference_inputs);
    for ((host_output, reference_output), spec) in host_outputs
        .iter()
        .chain(&second_outputs)
        .zip(reference_outputs.iter().chain(&second_reference_outputs))
        .zip(case.outputs().iter().cycle())
    {
        let actual = host.prepare_read(host_output).unwrap().run().await.unwrap();
        let expected = reference.read(reference_output).unwrap();
        assert_outputs_agree(spec.dtype(), &expected, &actual).unwrap();
        coverage.compared_reads += 1;
    }
    drop(host);
    fs::remove_file(path).unwrap();
}

fn run_reference_program(
    backend: &forja_cpu::CpuBackend,
    case: &forja_testing::program::ProgramCase,
    program: forja_core::program::ValidatedProgram,
    inputs: &[Tensor],
) -> (Vec<Tensor>, Vec<Tensor>) {
    let outputs = case
        .outputs()
        .iter()
        .map(|spec| allocate_reference_program_tensor(backend, spec, None))
        .collect::<Vec<_>>();
    let second_outputs = case
        .outputs()
        .iter()
        .map(|spec| allocate_reference_program_tensor(backend, spec, None))
        .collect::<Vec<_>>();
    let signature = KernelSignature::new(
        u8::try_from(case.shape().len()).unwrap(),
        case.inputs().iter().map(TensorSpec::dtype).collect(),
        case.outputs().iter().map(TensorSpec::dtype).collect(),
        0,
    );
    let prepared = prepare_program(backend, program, signature).unwrap();
    let input_refs = inputs.iter().collect::<Vec<_>>();
    let output_refs = outputs.iter().collect::<Vec<_>>();
    let second_output_refs = second_outputs.iter().collect::<Vec<_>>();
    let mut commands = CommandList::new();
    commands
        .dispatch_kernel(&prepared, &input_refs, &output_refs)
        .unwrap();
    commands
        .dispatch_kernel(&prepared, &input_refs, &second_output_refs)
        .unwrap();
    backend.submit(commands).unwrap().wait().unwrap();
    (outputs, second_outputs)
}

fn reuse_and_drop_kernel(
    host: &mut Host<forja_cpu::CpuBackend>,
    commands: &Resource<CommandListEntry>,
    kernel: Resource<KernelEntry>,
    program: Program,
    case: &forja_testing::program::ProgramCase,
    inputs: &[Resource<TensorEntry>],
    coverage: &mut ProgramCoverage,
) -> Vec<Resource<TensorEntry>> {
    let outputs = case
        .outputs()
        .iter()
        .map(|spec| allocate_host_program_tensor(host, spec, None))
        .collect::<Vec<_>>();
    host.dispatch_kernel(commands, &kernel, inputs, &outputs)
        .unwrap();
    let mut wrong_signature = wit_signature(case);
    wrong_signature.rank = wrong_signature.rank.checked_add(1).unwrap();
    let wrong_kernel = host
        .create_kernel(wit_program(program), wrong_signature)
        .unwrap();
    assert!(matches!(
        host.dispatch_kernel(commands, &wrong_kernel, inputs, &outputs),
        Err(compute::Error::OpSignature(_))
    ));
    coverage.signature_refusals += 1;
    host.drop_kernel(wrong_kernel).unwrap();
    host.drop_kernel(kernel).unwrap();
    outputs
}

fn allocate_host_program_tensor(
    host: &mut Host<forja_cpu::CpuBackend>,
    spec: &TensorSpec,
    bytes: Option<&[u8]>,
) -> Resource<TensorEntry> {
    let mut tensor = host
        .alloc(guest_dtype(spec.dtype()), spec.allocation_shape())
        .unwrap();
    if let Some(bytes) = bytes {
        host.write(&tensor, bytes).unwrap();
    }
    for view in spec.views() {
        tensor = host
            .view(&Resource::new_borrow(tensor.rep()), wit_view(view))
            .unwrap();
    }
    tensor
}

fn allocate_reference_program_tensor(
    backend: &forja_cpu::CpuBackend,
    spec: &TensorSpec,
    bytes: Option<&[u8]>,
) -> Tensor {
    let mut tensor = backend
        .alloc(spec.dtype(), spec.allocation_shape())
        .unwrap();
    if let Some(bytes) = bytes {
        backend.write(&tensor, bytes).unwrap();
    }
    for view in spec.views() {
        tensor = backend.view(&tensor, view.clone()).unwrap();
    }
    tensor
}

fn wit_view(view: &ViewOp) -> compute::ViewOp {
    match view {
        ViewOp::Slice(slices) => compute::ViewOp::Slice(
            slices
                .iter()
                .map(|&slice| compute::SliceSpec {
                    start: slice.start(),
                    len: slice.len(),
                    step: slice.step(),
                })
                .collect(),
        ),
        ViewOp::Reshape(shape) => compute::ViewOp::Reshape(shape.clone()),
        ViewOp::Permute(axes) => compute::ViewOp::Permute(axes.clone()),
        ViewOp::Broadcast(shape) => compute::ViewOp::Broadcast(shape.clone()),
    }
}

fn program_weight_file(spec: &TensorSpec) -> std::path::PathBuf {
    let byte_len =
        element_count(spec.allocation_shape()) * usize::try_from(spec.dtype().byte_size()).unwrap();
    let dtype = match spec.dtype() {
        DType::F32 => "F32",
        DType::F16 => "F16",
        DType::BF16 => "BF16",
        DType::I32 | DType::U32 => unreachable!("program outputs are floating point"),
    };
    let shape = spec.allocation_shape();
    let mut header = format!(
        "{{\"value\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[0,{byte_len}]}}}}"
    )
    .into_bytes();
    while !(header.len() + 8).is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.resize(bytes.len() + byte_len, 0);
    let path = std::env::temp_dir().join(format!(
        "forja-program-fuzz-{}-{}",
        std::process::id(),
        NEXT_PROGRAM_WEIGHT.fetch_add(1, Ordering::Relaxed)
    ));
    fs::write(&path, bytes).unwrap();
    path
}

fn wit_program(program: Program) -> compute::ProgramSource {
    compute::ProgramSource {
        kind: match program.kind {
            ProgramKind::Map => compute::ProgramKind::Map,
            ProgramKind::Row => compute::ProgramKind::Row,
        },
        insts: program.insts.into_iter().map(wit_inst).collect(),
        outputs: program.outputs,
    }
}

fn wit_signature(case: &forja_testing::program::ProgramCase) -> compute::KernelSignature {
    compute::KernelSignature {
        rank: u8::try_from(case.shape().len()).unwrap(),
        inputs: case
            .inputs()
            .iter()
            .map(|spec| guest_dtype(spec.dtype()))
            .collect(),
        outputs: case
            .outputs()
            .iter()
            .map(|spec| guest_dtype(spec.dtype()))
            .collect(),
        scalars: 0,
    }
}

fn wit_inst(inst: Inst) -> compute::Inst {
    match inst {
        Inst::Input(slot) => compute::Inst::Input(slot),
        Inst::Const(value) => compute::Inst::Const(value),
        Inst::Index(axis) => compute::Inst::Index(axis),
        Inst::Extent(axis) => compute::Inst::Extent(axis),
        Inst::Unary(op, value) => compute::Inst::Unary((wit_unop(op), value)),
        Inst::Binary(op, left, right) => compute::Inst::Binary((wit_binop(op), left, right)),
        Inst::Select(condition, accepted, rejected) => {
            compute::Inst::Select((condition, accepted, rejected))
        }
        Inst::Cast(to, value) => compute::Inst::Cast((wit_value_type(to), value)),
        Inst::Reduce(op, value) => compute::Inst::Reduce((wit_redop(op), value)),
    }
}

forja_program_conversions::program_op_conversions! {
    fn wit_unop(UnOp => WitUnOp);
    fn wit_binop(BinOp => WitBinOp);
    fn wit_redop(RedOp => WitRedOp);
}

forja_program_conversions::program_value_type_conversion! {
    fn wit_value_type(ValueType => WitValueType);
}

#[tokio::test(flavor = "multi_thread")]
async fn fuzzed_cpu_wit_sequences_match_the_reference() {
    let cases = 16;
    run_cases::<forja_cpu::CpuBackend, _>(cases, 64, forja_cpu::CpuBackend::new)
        .await
        .assert_sufficient(cases);
}

#[tokio::test(flavor = "multi_thread")]
async fn fuzzed_wit_programs_match_the_direct_interpreter() {
    let cases = 32;
    let mut coverage = ProgramCoverage::default();
    for seed in 1..=32 {
        run_program_case(seed, &mut coverage).await;
    }
    coverage.assert_sufficient(cases);
}

#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
async fn metal_tensor_smoke_fuzzed_wit_sequences_match_cpu() {
    let cases = 4;
    run_cases::<forja_metal::MetalBackend, _>(cases, 32, || {
        forja_metal::MetalBackend::new().unwrap()
    })
    .await
    .assert_sufficient(cases);
}
