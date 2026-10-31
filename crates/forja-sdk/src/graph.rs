use std::{
    cell::RefCell,
    collections::VecDeque,
    ops::{Add, RangeInclusive},
    rc::Rc,
    sync::atomic::{AtomicU64, Ordering},
};

use crate::{Element, Result, Tensor, sys};

thread_local! {
    static CURRENT: RefCell<Option<Recording>> = const { RefCell::new(None) };
    static KERNELS: RefCell<VecDeque<CachedKernel>> = const { RefCell::new(VecDeque::new()) };
    static EPILOGUES: RefCell<VecDeque<CachedEpilogue>> = const { RefCell::new(VecDeque::new()) };
}

static NEXT_PARAM_ID: AtomicU64 = AtomicU64::new(0);

enum Recording {
    Lazy(sys::Commands),
    Capture {
        commands: sys::Commands,
        params: sys::Params,
        ids: Vec<u64>,
    },
}

/// A declared scalar parameter accepted by a captured graph.
pub struct Param {
    id: u64,
    range: RangeInclusive<u32>,
}

impl Param {
    /// Declares one nonempty inclusive parameter range.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty range or exhausted process identities.
    pub fn new(range: RangeInclusive<u32>) -> Result<Self> {
        if range.is_empty() {
            return Err(crate::Error::new("parameter range cannot be empty"));
        }
        let id = NEXT_PARAM_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| crate::Error::new("parameter identities exhausted"))?;
        Ok(Self { id, range })
    }

    /// Traces this parameter at one concrete value.
    #[must_use]
    pub fn at(&self, value: u32) -> Pos {
        Pos {
            id: self.id,
            range: self.range.clone(),
            value,
        }
    }

    pub(crate) const fn id(&self) -> u64 {
        self.id
    }

    pub(crate) const fn range(&self) -> &RangeInclusive<u32> {
        &self.range
    }
}

/// A traced position carrying both its parameter identity and concrete value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Pos {
    id: u64,
    range: RangeInclusive<u32>,
    value: u32,
}

/// A concrete or affine tensor dimension.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Dim {
    id: Option<u64>,
    range: Option<RangeInclusive<u32>>,
    scale: u32,
    offset: u32,
    value: u32,
}

impl Dim {
    /// Returns the concrete value used for the current trace or lazy operation.
    #[must_use]
    pub const fn value(&self) -> u32 {
        self.value
    }

    pub(crate) const fn is_symbolic(&self) -> bool {
        self.id.is_some()
    }

    /// Adds a constant offset with checked arithmetic.
    ///
    /// # Errors
    ///
    /// Returns an error when either the trace value or affine offset overflows.
    pub fn checked_add(mut self, offset: u32) -> Result<Self> {
        self.offset = self
            .offset
            .checked_add(offset)
            .ok_or_else(|| crate::Error::new("affine offset overflowed"))?;
        self.value = self
            .value
            .checked_add(offset)
            .ok_or_else(|| crate::Error::new("traced dimension overflowed"))?;
        Ok(self)
    }

    /// Multiplies this affine dimension by a constant.
    ///
    /// # Errors
    ///
    /// Returns an error when the affine scale, offset, or trace value overflows.
    pub fn checked_mul(mut self, factor: u32) -> Result<Self> {
        self.scale = self
            .scale
            .checked_mul(factor)
            .ok_or_else(|| crate::Error::new("affine scale overflowed"))?;
        self.offset = self
            .offset
            .checked_mul(factor)
            .ok_or_else(|| crate::Error::new("affine offset overflowed"))?;
        self.value = self
            .value
            .checked_mul(factor)
            .ok_or_else(|| crate::Error::new("traced dimension overflowed"))?;
        Ok(self)
    }
}

impl From<u32> for Dim {
    fn from(value: u32) -> Self {
        Self {
            id: None,
            range: None,
            scale: 0,
            offset: value,
            value,
        }
    }
}

impl From<Pos> for Dim {
    fn from(position: Pos) -> Self {
        Self {
            id: Some(position.id),
            range: Some(position.range),
            scale: 1,
            offset: 0,
            value: position.value,
        }
    }
}

impl From<&Dim> for Dim {
    fn from(dimension: &Dim) -> Self {
        dimension.clone()
    }
}

impl Add<u32> for Pos {
    type Output = Result<Dim>;

    fn add(self, offset: u32) -> Self::Output {
        Dim::from(self).checked_add(offset)
    }
}

/// A captured graph and the values produced while tracing it.
pub struct Graph<T> {
    graph: sys::Graph,
    result: T,
}

/// Values whose tensor storage remains valid after a captured graph replays.
pub trait GraphResult {
    #[doc(hidden)]
    fn protect_graph_result(&self) -> Result<()>;
}

impl<T: Element> GraphResult for Tensor<T> {
    fn protect_graph_result(&self) -> Result<()> {
        CURRENT.with(|current| {
            let mut current = current.borrow_mut();
            let Some(Recording::Capture { commands, .. }) = current.as_mut() else {
                return Err(crate::Error::new("graph result requires active capture"));
            };
            sys::graph_output(commands, self.handle())
        })
    }
}

impl GraphResult for () {
    fn protect_graph_result(&self) -> Result<()> {
        Ok(())
    }
}

impl<T: GraphResult> GraphResult for Vec<T> {
    fn protect_graph_result(&self) -> Result<()> {
        self.iter().try_for_each(GraphResult::protect_graph_result)
    }
}

impl<T: GraphResult> GraphResult for Option<T> {
    fn protect_graph_result(&self) -> Result<()> {
        self.as_ref()
            .map_or(Ok(()), GraphResult::protect_graph_result)
    }
}

impl<T: Element> GraphResult for T {
    fn protect_graph_result(&self) -> Result<()> {
        Ok(())
    }
}

impl<A: GraphResult, B: GraphResult> GraphResult for (A, B) {
    fn protect_graph_result(&self) -> Result<()> {
        self.0.protect_graph_result()?;
        self.1.protect_graph_result()
    }
}

impl<T> Graph<T> {
    /// Replays the captured graph with concrete parameter values.
    ///
    /// # Errors
    ///
    /// Returns a host validation or execution error.
    pub fn replay(&self, values: &[u32]) -> Result<()> {
        sys::replay(&self.graph, values)
    }

    /// Returns the values produced while the graph was traced.
    #[must_use]
    pub const fn result(&self) -> &T {
        &self.result
    }
}

/// Captures lazy tensor operations into a replayable graph.
///
/// # Errors
///
/// Returns an error for nested capture, pending lazy work, a duplicate parameter,
/// a trace failure, or a host-refused graph.
pub fn capture<T: GraphResult>(
    params: &[&Param],
    trace: impl FnOnce() -> Result<T>,
) -> Result<Graph<T>> {
    let mut ids = Vec::with_capacity(params.len());
    for parameter in params {
        if ids.contains(&parameter.id) {
            return Err(crate::Error::new("capture parameters must be unique"));
        }
        ids.push(parameter.id);
    }
    let ranges = params
        .iter()
        .map(|parameter| (*parameter.range.start(), *parameter.range.end()))
        .collect::<Vec<_>>();
    let recording = Recording::Capture {
        commands: sys::command_list()?,
        params: sys::params(&ranges)?,
        ids,
    };
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_some() {
            return Err(crate::Error::new(
                "capture requires the lazy graph to be empty",
            ));
        }
        *current = Some(recording);
        drop(current);

        let result = trace();
        let protection = result
            .as_ref()
            .map_or(Ok(()), GraphResult::protect_graph_result);
        let recording = CURRENT
            .with(|current| current.borrow_mut().take())
            .ok_or_else(|| crate::Error::new("capture recording was lost"))?;
        let commands = match recording {
            Recording::Capture { commands, .. } => commands,
            Recording::Lazy(_) => return Err(crate::Error::new("capture recording was replaced")),
        };
        let result = result?;
        protection?;
        Ok(Graph {
            graph: sys::create_graph(commands)?,
            result,
        })
    })
}

pub(crate) fn alloc_output(dtype: u8, shape: &[u32]) -> Result<sys::Handle> {
    CURRENT.with(|current| match current.borrow().as_ref() {
        Some(Recording::Capture { commands, .. }) => sys::alloc_scratch(commands, dtype, shape),
        _ => sys::alloc(dtype, shape),
    })
}

pub(crate) fn affine(dim: &Dim) -> Result<sys::Affine> {
    let Some(id) = dim.id else {
        return Ok(sys::Affine {
            param: None,
            scale: 0,
            offset: dim.offset,
        });
    };
    parameter_trace_value(dim)?;
    CURRENT.with(|current| {
        let current = current.borrow();
        let Some(Recording::Capture { ids, .. }) = current.as_ref() else {
            return Err(crate::Error::new(
                "symbolic dimensions can only be used during capture",
            ));
        };
        let slot = ids
            .iter()
            .position(|candidate| *candidate == id)
            .ok_or_else(|| crate::Error::new("dimension parameter is not part of this capture"))?;
        Ok(sys::Affine {
            param: Some(
                u8::try_from(slot).map_err(|_| crate::Error::new("parameter slot exceeds u8"))?,
            ),
            scale: dim.scale,
            offset: dim.offset,
        })
    })
}

fn parameter_trace_value(dim: &Dim) -> Result<u32> {
    let trace_value = dim
        .value
        .checked_sub(dim.offset)
        .ok_or_else(|| crate::Error::new("affine offset exceeds the trace value"))?;
    if dim
        .range
        .as_ref()
        .is_none_or(|range| !range.contains(&trace_value))
    {
        return Err(crate::Error::new(
            "trace value is outside its parameter range",
        ));
    }
    Ok(trace_value)
}

pub(crate) fn view_param(tensor: &sys::Handle, slices: &[sys::ParamSlice]) -> Result<sys::Handle> {
    CURRENT.with(|current| {
        let current = current.borrow();
        let Some(Recording::Capture { params, .. }) = current.as_ref() else {
            return Err(crate::Error::new(
                "parameterized views can only be used during capture",
            ));
        };
        sys::view_param(tensor, params, slices)
    })
}

pub(crate) fn refuse_during_capture(operation: &str) -> Result<()> {
    CURRENT.with(|current| {
        if matches!(current.borrow().as_ref(), Some(Recording::Capture { .. })) {
            Err(crate::Error::new(format!(
                "{operation} is not allowed during graph capture"
            )))
        } else {
            Ok(())
        }
    })
}

const KERNEL_CACHE_CAPACITY: usize = 8;

#[derive(Eq, PartialEq)]
struct KernelKey {
    source: Vec<u8>,
    rank: u8,
    inputs: Vec<sys::DType>,
    outputs: Vec<sys::DType>,
}

struct CachedKernel {
    key: KernelKey,
    kernel: Rc<sys::Kernel>,
}

#[derive(Eq, PartialEq)]
struct EpilogueKey {
    source: Vec<u8>,
    accumulator_count: u8,
    rank: u8,
    extras: Vec<sys::DType>,
    output: sys::DType,
}

struct CachedEpilogue {
    key: EpilogueKey,
    epilogue: Rc<sys::Epilogue>,
}

pub(crate) fn record(
    operation: sys::Op,
    inputs: &[&sys::Handle],
    output: &sys::Handle,
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            *current = Some(Recording::Lazy(sys::command_list()?));
        }
        let Some(Recording::Lazy(commands) | Recording::Capture { commands, .. }) =
            current.as_mut()
        else {
            return Err(crate::Error::new("current graph was not initialized"));
        };
        sys::dispatch(commands, operation, inputs, output)
    })
}

pub(crate) fn record_many(
    operation: sys::Op,
    inputs: &[&sys::Handle],
    outputs: &[&sys::Handle],
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            *current = Some(Recording::Lazy(sys::command_list()?));
        }
        let Some(Recording::Lazy(commands) | Recording::Capture { commands, .. }) =
            current.as_mut()
        else {
            return Err(crate::Error::new("current graph was not initialized"));
        };
        sys::dispatch_many(commands, operation, inputs, outputs)
    })
}

#[cfg(target_family = "wasm")]
pub(crate) fn capture_import<T>(
    operation: impl FnOnce(&mut sys::Commands) -> Result<T>,
) -> Result<Option<T>> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if let Some(Recording::Capture { commands, .. }) = current.as_mut() {
            operation(commands).map(Some)
        } else {
            Ok(None)
        }
    })
}

#[cfg(target_family = "wasm")]
pub(crate) fn record_import(
    operation: impl FnOnce(&mut sys::Commands) -> Result<()>,
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            *current = Some(Recording::Lazy(sys::command_list()?));
        }
        let Some(Recording::Lazy(commands) | Recording::Capture { commands, .. }) =
            current.as_mut()
        else {
            return Err(crate::Error::new("current graph was not initialized"));
        };
        operation(commands)
    })
}

#[cfg(target_family = "wasm")]
pub(crate) fn record_import_variant(
    selection: crate::target::metal::VariantChoice<'_>,
    fixed: impl FnOnce(&mut sys::Commands, &str) -> Result<()>,
    rule: impl FnOnce(&mut sys::Commands, &sys::Params, u8, &[sys::VariantArm]) -> Result<()>,
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            if matches!(selection, crate::target::metal::VariantChoice::Rule(_)) {
                return Err(crate::Error::new(
                    "Metal variant rules can only be used during capture",
                ));
            }
            *current = Some(Recording::Lazy(sys::command_list()?));
        }
        match (current.as_mut(), selection) {
            (
                Some(Recording::Lazy(commands) | Recording::Capture { commands, .. }),
                crate::target::metal::VariantChoice::Variant(variant),
            ) => fixed(commands, variant.name()),
            (
                Some(Recording::Capture {
                    commands,
                    params,
                    ids,
                }),
                crate::target::metal::VariantChoice::Rule(selection),
            ) => {
                let parameter = ids
                    .iter()
                    .position(|candidate| *candidate == selection.parameter_id)
                    .and_then(|slot| u8::try_from(slot).ok())
                    .ok_or_else(|| {
                        crate::Error::new("variant rule parameter is not part of this capture")
                    })?;
                rule(commands, params, parameter, &selection.arms)
            }
            (Some(Recording::Lazy(_)), crate::target::metal::VariantChoice::Rule(_)) => Err(
                crate::Error::new("Metal variant rules can only be used during capture"),
            ),
            (None, _) => Err(crate::Error::new("current graph was not initialized")),
        }
    })
}

/// Sets engine-owned call-site and layer attribution for subsequent operations.
///
/// Attribution is retained by captured graphs and does not affect tensor execution.
///
/// # Errors
/// Returns an error for invalid names or if command-list creation fails.
pub fn profile_scope(site: &str, layer_type: &str, layer: Option<u32>) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            *current = Some(Recording::Lazy(sys::command_list()?));
        }
        match current.as_mut() {
            Some(Recording::Lazy(commands) | Recording::Capture { commands, .. }) => {
                sys::set_profile_scope(commands, site, layer_type, layer)
            }
            None => Err(crate::Error::new("current graph was not initialized")),
        }
    })
}

pub(crate) fn record_variant(
    operation: sys::Op,
    inputs: &[&sys::Handle],
    output: &sys::Handle,
    selection: crate::target::metal::VariantChoice<'_>,
) -> Result<()> {
    record_variant_many(operation, inputs, &[output], selection)
}

pub(crate) fn record_variant_many(
    operation: sys::Op,
    inputs: &[&sys::Handle],
    outputs: &[&sys::Handle],
    selection: crate::target::metal::VariantChoice<'_>,
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            if matches!(selection, crate::target::metal::VariantChoice::Rule(_)) {
                return Err(crate::Error::new(
                    "Metal variant rules can only be used during capture",
                ));
            }
            *current = Some(Recording::Lazy(sys::command_list()?));
        }
        match (current.as_mut(), selection) {
            (
                Some(Recording::Lazy(commands) | Recording::Capture { commands, .. }),
                crate::target::metal::VariantChoice::Variant(variant),
            ) => sys::dispatch_variant(commands, operation, inputs, outputs, variant.name()),
            (
                Some(Recording::Capture {
                    commands,
                    params,
                    ids,
                }),
                crate::target::metal::VariantChoice::Rule(rule),
            ) => {
                let parameter = ids
                    .iter()
                    .position(|candidate| *candidate == rule.parameter_id)
                    .and_then(|slot| u8::try_from(slot).ok())
                    .ok_or_else(|| {
                        crate::Error::new("variant rule parameter is not part of this capture")
                    })?;
                sys::dispatch_variant_rule(
                    commands, params, parameter, &rule.arms, operation, inputs, outputs,
                )
            }
            (Some(Recording::Lazy(_)), crate::target::metal::VariantChoice::Rule(_)) => Err(
                crate::Error::new("Metal variant rules can only be used during capture"),
            ),
            (None, _) => Err(crate::Error::new("current graph was not initialized")),
        }
    })
}

pub(crate) fn record_program(
    program: sys::Program,
    rank: u8,
    input_dtypes: Vec<sys::DType>,
    output_dtypes: Vec<sys::DType>,
    inputs: &[&sys::Handle],
    outputs: &[&sys::Handle],
) -> Result<()> {
    let kernel = cached_kernel(program, rank, input_dtypes, output_dtypes)?;
    record_kernel(&kernel, inputs, outputs)
}

pub(crate) fn record_kernel(
    kernel: &sys::Kernel,
    inputs: &[&sys::Handle],
    outputs: &[&sys::Handle],
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            *current = Some(Recording::Lazy(sys::command_list()?));
        }
        let Some(Recording::Lazy(commands) | Recording::Capture { commands, .. }) =
            current.as_mut()
        else {
            return Err(crate::Error::new("current graph was not initialized"));
        };
        sys::dispatch_kernel(commands, kernel, inputs, outputs)
    })
}

pub(crate) fn record_epilogue(
    operation: sys::Op,
    inputs: &[&sys::Handle],
    output: &sys::Handle,
    epilogue: &sys::Epilogue,
    extras: &[&sys::Handle],
    selection: Option<crate::target::metal::VariantChoice<'_>>,
) -> Result<()> {
    CURRENT.with(|current| {
        let mut current = current.borrow_mut();
        if current.is_none() {
            if matches!(
                selection,
                Some(crate::target::metal::VariantChoice::Rule(_))
            ) {
                return Err(crate::Error::new(
                    "Metal variant rules can only be used during capture",
                ));
            }
            *current = Some(Recording::Lazy(sys::command_list()?));
        }
        match (current.as_mut(), selection) {
            (Some(Recording::Lazy(commands) | Recording::Capture { commands, .. }), None) => {
                sys::dispatch_epilogue(commands, operation, inputs, output, epilogue, extras)
            }
            (
                Some(Recording::Lazy(commands) | Recording::Capture { commands, .. }),
                Some(crate::target::metal::VariantChoice::Variant(variant)),
            ) => {
                sys::dispatch_epilogue(commands, operation, inputs, output, epilogue, extras)?;
                sys::select_epilogue_variant(commands, variant.name())
            }
            (
                Some(Recording::Capture {
                    commands,
                    params,
                    ids,
                }),
                Some(crate::target::metal::VariantChoice::Rule(rule)),
            ) => {
                let parameter = ids
                    .iter()
                    .position(|candidate| *candidate == rule.parameter_id)
                    .and_then(|slot| u8::try_from(slot).ok())
                    .ok_or_else(|| {
                        crate::Error::new("variant rule parameter is not part of this capture")
                    })?;
                sys::dispatch_epilogue(commands, operation, inputs, output, epilogue, extras)?;
                sys::select_epilogue_variant_rule(commands, params, parameter, &rule.arms)
            }
            (Some(Recording::Lazy(_)), Some(crate::target::metal::VariantChoice::Rule(_))) => Err(
                crate::Error::new("Metal variant rules can only be used during capture"),
            ),
            (None, _) => Err(crate::Error::new("current graph was not initialized")),
        }
    })
}

pub(crate) fn cached_epilogue(
    program: sys::Program,
    accumulator_count: u8,
    rank: u8,
    extras: Vec<sys::DType>,
    output: sys::DType,
) -> Result<Rc<sys::Epilogue>> {
    let key = EpilogueKey {
        source: program_bytes(&program),
        accumulator_count,
        rank,
        extras,
        output,
    };
    EPILOGUES.with(|epilogues| {
        let mut epilogues = epilogues.borrow_mut();
        if let Some(index) = epilogues.iter().position(|entry| entry.key == key)
            && let Some(entry) = epilogues.remove(index)
        {
            let epilogue = Rc::clone(&entry.epilogue);
            epilogues.push_back(entry);
            return Ok(epilogue);
        }
        if epilogues.len() == KERNEL_CACHE_CAPACITY {
            epilogues.pop_front();
        }
        let epilogue = Rc::new(sys::create_epilogue(
            program,
            accumulator_count,
            rank,
            &key.extras,
            output,
        )?);
        epilogues.push_back(CachedEpilogue {
            key,
            epilogue: Rc::clone(&epilogue),
        });
        Ok(epilogue)
    })
}

fn cached_kernel(
    program: sys::Program,
    rank: u8,
    inputs: Vec<sys::DType>,
    outputs: Vec<sys::DType>,
) -> Result<Rc<sys::Kernel>> {
    let key = KernelKey {
        source: program_bytes(&program),
        rank,
        inputs,
        outputs,
    };
    KERNELS.with(|kernels| {
        let mut kernels = kernels.borrow_mut();
        if let Some(index) = kernels.iter().position(|entry| entry.key == key)
            && let Some(entry) = kernels.remove(index)
        {
            let kernel = Rc::clone(&entry.kernel);
            kernels.push_back(entry);
            return Ok(kernel);
        }
        if kernels.len() == KERNEL_CACHE_CAPACITY {
            kernels.pop_front();
        }
        let kernel = Rc::new(sys::create_kernel(
            program,
            rank,
            &key.inputs,
            &key.outputs,
        )?);
        kernels.push_back(CachedKernel {
            key,
            kernel: Rc::clone(&kernel),
        });
        Ok(kernel)
    })
}

fn program_bytes(program: &sys::Program) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.push(match program.kind {
        crate::program::ProgramKind::Map => 0,
        crate::program::ProgramKind::Row => 1,
    });
    bytes.extend(program.instructions.len().to_le_bytes());
    for instruction in &program.instructions {
        encode_instruction(&mut bytes, *instruction);
    }
    bytes.push(0xff);
    for &(slot, value) in &program.outputs {
        bytes.extend(slot.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

#[allow(clippy::too_many_lines)]
fn encode_instruction(bytes: &mut Vec<u8>, instruction: sys::ProgramInst) {
    use sys::ProgramInst;
    match instruction {
        ProgramInst::Input(slot) => encode_one(bytes, 0, slot),
        ProgramInst::Constant(value) => encode_one(bytes, 1, value.to_bits()),
        ProgramInst::Index(axis) => encode_axis(bytes, 2, axis),
        ProgramInst::Extent(axis) => encode_axis(bytes, 3, axis),
        ProgramInst::Unary(op, value) => {
            bytes.extend([4, op as u8]);
            bytes.extend(value.to_le_bytes());
        }
        ProgramInst::Binary(op, left, right) => {
            bytes.extend([5, op as u8]);
            bytes.extend(left.to_le_bytes());
            bytes.extend(right.to_le_bytes());
        }
        ProgramInst::Select(condition, accepted, rejected) => {
            bytes.push(6);
            bytes.extend(condition.to_le_bytes());
            bytes.extend(accepted.to_le_bytes());
            bytes.extend(rejected.to_le_bytes());
        }
        ProgramInst::Cast(to, value) => {
            bytes.extend([
                7,
                match to {
                    crate::program::ValueType::F32 => 0,
                    crate::program::ValueType::U32 => 1,
                    crate::program::ValueType::Bool => 2,
                },
            ]);
            bytes.extend(value.to_le_bytes());
        }
        ProgramInst::Reduce(op, value) => {
            bytes.extend([8, op as u8]);
            bytes.extend(value.to_le_bytes());
        }
    }
}

fn encode_one(bytes: &mut Vec<u8>, tag: u8, value: u32) {
    bytes.push(tag);
    bytes.extend(value.to_le_bytes());
}

fn encode_axis(bytes: &mut Vec<u8>, tag: u8, axis: u8) {
    bytes.extend([tag, axis]);
}

#[cfg(feature = "native")]
pub(crate) fn clear_kernel_cache() {
    KERNELS.with(|kernels| kernels.borrow_mut().clear());
    EPILOGUES.with(|epilogues| epilogues.borrow_mut().clear());
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;
    use crate::program::ProgramKind;

    fn program(value: f32) -> sys::Program {
        sys::Program {
            kind: ProgramKind::Map,
            instructions: vec![
                sys::ProgramInst::Input(0),
                sys::ProgramInst::Constant(value),
                sys::ProgramInst::Binary(crate::program::BinaryOp::Mul, 0, 1),
            ],
            outputs: vec![(0, 2)],
        }
    }

    #[test]
    fn program_cache_reuses_handles_and_evicts_at_its_bound() {
        clear_kernel_cache();
        let first = cached_kernel(
            program(1.0),
            1,
            vec![sys::DType::F32],
            vec![sys::DType::F32],
        )
        .unwrap();
        let repeated = cached_kernel(
            program(1.0),
            1,
            vec![sys::DType::F32],
            vec![sys::DType::F32],
        )
        .unwrap();
        assert!(Rc::ptr_eq(&first, &repeated));

        for value in 2_u16..=9 {
            cached_kernel(
                program(f32::from(value)),
                1,
                vec![sys::DType::F32],
                vec![sys::DType::F32],
            )
            .unwrap();
        }
        KERNELS.with(|kernels| assert_eq!(kernels.borrow().len(), KERNEL_CACHE_CAPACITY));
    }

    #[test]
    fn epilogue_cache_keys_accumulator_count_and_signature() {
        clear_kernel_cache();
        let first = cached_epilogue(program(1.0), 1, 2, vec![], sys::DType::F32).unwrap();
        let repeated = cached_epilogue(program(1.0), 1, 2, vec![], sys::DType::F32).unwrap();
        let different = cached_epilogue(
            sys::Program {
                kind: ProgramKind::Map,
                instructions: vec![
                    sys::ProgramInst::Input(0),
                    sys::ProgramInst::Input(1),
                    sys::ProgramInst::Binary(crate::program::BinaryOp::Add, 0, 1),
                ],
                outputs: vec![(0, 2)],
            },
            2,
            2,
            vec![],
            sys::DType::F32,
        )
        .unwrap();

        assert!(Rc::ptr_eq(&first, &repeated));
        assert!(!Rc::ptr_eq(&first, &different));
    }

    #[test]
    fn affine_offset_preserves_the_trace_value_at_the_range_end() {
        let parameter = Param::new(0..=6).unwrap();
        let dimension = (parameter.at(6) + 1).unwrap();
        assert_eq!(parameter_trace_value(&dimension).unwrap(), 6);
    }
}

/// Submits all operations recorded by the current thread.
///
/// # Errors
///
/// Returns a host validation or execution error.
pub fn eval() -> Result<()> {
    refuse_during_capture("eval")?;
    CURRENT.with(|current| match current.borrow_mut().take() {
        Some(Recording::Lazy(commands)) => sys::submit(commands),
        Some(Recording::Capture { .. }) => Err(crate::Error::new("capture recording was lost")),
        None => Ok(()),
    })
}
