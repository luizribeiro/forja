use super::{StateRecording, StateSequenceHandle, StateStorage};
use crate::{Host, Limits, compute};
use forja_core::{
    Backend, BackendError, BufferId, CommandList, DType, Layout, PagePoolError, StateLayout,
    StatePlane, Submission, Tensor, ViewOp,
    program::{KernelSignature, ValidatedProgram},
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering},
    },
    time::Duration,
};

#[derive(Default)]
struct Control {
    waits: AtomicUsize,
    failure: AtomicU8,
    timeout_ns: AtomicU64,
    gpu_ns: AtomicU64,
}

struct ControlledBackend {
    cpu: forja_cpu::CpuBackend,
    control: Arc<Control>,
}

struct ControlledSubmission {
    inner: forja_cpu::CpuSubmission,
    control: Arc<Control>,
    failure: u8,
}

impl Submission for ControlledSubmission {
    fn wait(&self) -> Result<(), BackendError> {
        self.control.waits.fetch_add(1, Ordering::AcqRel);
        match self.failure {
            1 => Err(BackendError::Timeout),
            2 => Err(BackendError::ExecutionFailed),
            _ => self.inner.wait(),
        }
    }
    fn wait_timeout(&self, timeout: Duration) -> Result<(), BackendError> {
        self.control.timeout_ns.store(
            u64::try_from(timeout.as_nanos()).unwrap(),
            Ordering::Release,
        );
        self.wait()
    }
    fn gpu_time(&self) -> Option<Duration> {
        match self.control.gpu_ns.load(Ordering::Acquire) {
            0 => None,
            nanos => Some(Duration::from_nanos(nanos)),
        }
    }
}

impl Backend for ControlledBackend {
    type Submission = ControlledSubmission;
    type ProgramHandle = ();
    fn prepare_program(
        &self,
        program: &ValidatedProgram,
        signature: &KernelSignature,
    ) -> Result<(), BackendError> {
        self.cpu.prepare_program(program, signature)
    }
    fn alloc(&self, dtype: DType, shape: &[u32]) -> Result<Tensor, BackendError> {
        self.cpu.alloc(dtype, shape)
    }
    fn tensor(&self, buffer: BufferId, layout: Layout) -> Result<Tensor, BackendError> {
        self.cpu.tensor(buffer, layout)
    }
    fn view(&self, tensor: &Tensor, op: ViewOp) -> Result<Tensor, BackendError> {
        self.cpu.view(tensor, op)
    }
    fn write(&self, tensor: &Tensor, bytes: &[u8]) -> Result<(), BackendError> {
        self.cpu.write(tensor, bytes)
    }
    fn read(&self, tensor: &Tensor) -> Result<Vec<u8>, BackendError> {
        self.cpu.read(tensor)
    }
    fn release(&self, tensor: &Tensor) -> Result<(), BackendError> {
        self.cpu.release(tensor)
    }
    fn submit(&self, commands: CommandList) -> Result<Self::Submission, BackendError> {
        Ok(ControlledSubmission {
            inner: self.cpu.submit(commands)?,
            control: Arc::clone(&self.control),
            failure: self.control.failure.load(Ordering::Acquire),
        })
    }
}

type Setup = (
    Arc<Control>,
    Arc<StateStorage<ControlledBackend>>,
    StateSequenceHandle<ControlledBackend>,
);
fn setup() -> Setup {
    let control = Arc::new(Control::default());
    let host = Host::new(
        ControlledBackend {
            cpu: forja_cpu::CpuBackend::new(),
            control: Arc::clone(&control),
        },
        Limits::new(65536, 8, 4096, 128, 4096),
    );
    let layout = StateLayout::new(vec![StatePlane::new(DType::F32, vec![1, 1]).unwrap()]).unwrap();
    let storage = host.create_state_storage(layout, 1).unwrap();
    let sequence = storage.reserve().unwrap();
    (control, storage, sequence)
}

fn recording(
    storage: &Arc<StateStorage<ControlledBackend>>,
    sequence: &StateSequenceHandle<ControlledBackend>,
    serial: u64,
) -> StateRecording<ControlledBackend> {
    let tx = sequence.begin(serial).unwrap();
    let mut recording = storage.record(sequence, tx).unwrap();
    let x = storage.backend.alloc(DType::F32, &[1, 1]).unwrap();
    let weights = storage.backend.alloc(DType::F32, &[2, 1]).unwrap();
    let y = storage.backend.alloc(DType::F32, &[1, 1]).unwrap();
    recording.causal_convolution(0, &x, &weights, &y).unwrap();
    recording
}

#[test]
fn timeout_waits_once_and_restores_the_committed_checkpoint() {
    let (control, storage, sequence) = setup();
    recording(&storage, &sequence, 1)
        .submit()
        .unwrap()
        .finish()
        .unwrap();
    control.failure.store(1, Ordering::Release);
    let error = recording(&storage, &sequence, 2)
        .submit()
        .unwrap()
        .finish()
        .unwrap_err();
    assert!(matches!(error, compute::Error::Quota(_)));
    assert_eq!(control.waits.load(Ordering::Acquire), 2);
    let retry = sequence.begin(3).unwrap();
    assert_eq!(retry.input_version(), 1);
    assert_eq!(
        sequence.lock().mark_submitted_at(&retry, 3),
        Err(PagePoolError::BankInFlight)
    );
    assert!(storage.record(&sequence, retry).is_err());
}

#[test]
fn early_drop_reconciles_once_including_timeout() {
    for failure in [0, 1] {
        let (control, storage, sequence) = setup();
        control.failure.store(failure, Ordering::Release);
        drop(recording(&storage, &sequence, 1).submit().unwrap());
        assert_eq!(control.waits.load(Ordering::Acquire), 1);
        assert_eq!(sequence.begin(2).unwrap().input_version(), 0);
    }
}

#[test]
fn a_secondary_finish_error_is_not_masked_by_execution_failure() {
    let (control, storage, sequence) = setup();
    let first = recording(&storage, &sequence, 1).submit().unwrap();
    control.failure.store(2, Ordering::Release);
    let second = recording(&storage, &sequence, 2).submit().unwrap();
    assert!(matches!(
        second.finish(),
        Err(compute::Error::InvalidHandle(_))
    ));
    drop(first);
    assert_eq!(control.waits.load(Ordering::Acquire), 2);
    assert_eq!(sequence.begin(3).unwrap().input_version(), 0);
}

#[test]
fn submission_owns_its_sequence_and_backing_through_early_drop() {
    let (control, storage, sequence) = setup();
    let retained = Arc::downgrade(&storage);
    let submission = recording(&storage, &sequence, 1).submit().unwrap();
    drop(sequence);
    drop(storage);
    assert!(retained.upgrade().is_some());
    drop(submission);
    assert_eq!(control.waits.load(Ordering::Acquire), 1);
    assert!(retained.upgrade().is_none());
}

#[test]
fn state_completion_uses_the_configured_deadline_and_reports_gpu_time() {
    let (control, storage, sequence) = setup();
    control.gpu_ns.store(7, Ordering::Release);
    let time = recording(&storage, &sequence, 1)
        .submit()
        .unwrap()
        .finish_with_time()
        .unwrap();
    assert_eq!(time, Some(Duration::from_nanos(7)));
    assert_eq!(control.waits.load(Ordering::Acquire), 1);
    assert_eq!(
        control.timeout_ns.load(Ordering::Acquire),
        u64::try_from(storage.limits.submission_timeout.as_nanos()).unwrap()
    );
}
