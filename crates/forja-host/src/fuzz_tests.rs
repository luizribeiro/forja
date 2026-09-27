use std::time::Duration;

use forja_core::Backend;
use forja_testing::assert_outputs_agree;
use wasmtime::component::Resource;

use super::{
    CommandListEntry, Host, Limits, TensorEntry, bindings::l9o::gpu::compute, core_dtype,
    guest_error,
};

const FUZZ_LIMITS: Limits = Limits::new(8 * 1024 * 1024, 8, 1_000_000, 128, 8 * 1024 * 1024)
    .with_command_limits(16, 1_000_000_000)
    .with_gpu_limits(Duration::from_secs(60), Duration::MAX);

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
                q_start: 0,
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
            q_start: u32::try_from(values.index(40)).unwrap(),
        }),
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

#[tokio::test(flavor = "multi_thread")]
async fn fuzzed_cpu_wit_sequences_match_the_reference() {
    let cases = 16;
    run_cases::<forja_cpu::CpuBackend, _>(cases, 64, forja_cpu::CpuBackend::new)
        .await
        .assert_sufficient(cases);
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
