#[cfg(test)]
use super::CommandList;
use super::{
    ArgumentWriter, BackendError, Dispatch, DispatchEncoding, HashSet, MTL4CommandBuffer,
    MetalBackend, Op, ProfileTimestampGranularity, RefCell, ScratchScope,
};

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) struct Request {
    pub(super) name: String,
    pub(super) constants: Vec<(u32, u32)>,
}
thread_local! {
    static REQUESTS: RefCell<Option<Vec<Request>>> = const { RefCell::new(None) };
}
struct Recorder(Option<Vec<Request>>);
impl Recorder {
    fn enter() -> Self {
        Self(REQUESTS.replace(Some(Vec::new())))
    }
    fn finish(self) -> Vec<Request> {
        let requests = REQUESTS.replace(None).unwrap_or_default();
        drop(self);
        requests
    }
}
impl Drop for Recorder {
    fn drop(&mut self) {
        REQUESTS.set(self.0.take());
    }
}
pub(super) fn record(name: &str, constants: &[(u32, u32)]) {
    REQUESTS.with_borrow_mut(|requests| {
        if let Some(requests) = requests {
            requests.push(Request {
                name: name.to_owned(),
                constants: constants.to_vec(),
            });
        }
    });
}
pub(super) fn generic_constants(name: &str, constants: &[(u32, u32)]) -> Option<Vec<(u32, u32)>> {
    let mut generic = constants.to_vec();
    for (index, value) in &mut generic {
        if (name.starts_with("rms_norm_") && *index == 21)
            || (name.starts_with("state_") && matches!(*index, 227 | 228))
            || (name.contains("quantized") && *index == 224)
        {
            *value = 0;
        }
    }
    (generic != constants).then_some(generic)
}

fn alternatives(request: &Request, dispatch: &Dispatch, entries: &HashSet<String>) -> Vec<String> {
    const DENSE: [&str; 7] = [
        "gemv",
        "gemv_transposed",
        "column_gemv",
        "steel_gemm_64_64_16_2_2",
        "steel_gemm_64_64_16_1_2",
        "steel_gemm_64_32_32_2_2",
        "steel_gemm_32_64_16_1_2",
    ];
    let name = request.name.as_str();
    let names = match name {
        "copy_contiguous_f32x4" | "copy_contiguous" | "copy_strided" => {
            vec!["copy_contiguous_f32x4", "copy_contiguous", "copy_strided"]
        }
        "argmax_partials" | "argmax_partials_f32x4" => {
            vec!["argmax_partials", "argmax_partials_f32x4"]
        }
        "rms_norm_single" | "rms_norm_looped" => vec!["rms_norm_single", "rms_norm_looped"],
        "softmax_single" | "softmax_looped" => vec!["softmax_single", "softmax_looped"],
        _ if DENSE.contains(&name)
            && matches!(
                dispatch.op(),
                Op::Sdpa { .. } | Op::PagedSdpa { .. } | Op::PagedSdpaAppend { .. }
            ) =>
        {
            DENSE.to_vec()
        }
        _ => vec![name],
    };
    let mut names = names.into_iter().map(str::to_owned).collect::<Vec<_>>();
    if name.starts_with("mlx_sdpa_vector") && !name.contains("_gqa_") {
        let plain = name.replace("_f32", "");
        names.push(plain.clone());
        if dispatch
            .inputs()
            .iter()
            .take(3)
            .chain(std::iter::once(dispatch.output()))
            .all(|tensor| tensor.layout().dtype() == forja_core::DType::F32)
            && let Some((prefix, width)) = plain.rsplit_once('_')
        {
            names.push(format!("{prefix}_f32_{width}"));
        }
    }
    let family = if name.starts_with("paged_sdpa_2pass_1") {
        Some("paged_sdpa_2pass_1")
    } else if name.starts_with("paged_sdpa_2pass_2") {
        Some("paged_sdpa_2pass_2")
    } else if name.starts_with("mlx_sdpa_vector_2pass_1") {
        Some("mlx_sdpa_vector_2pass_1")
    } else if name.starts_with("mlx_sdpa_vector_2pass_2") {
        Some("mlx_sdpa_vector_2pass_2")
    } else if name.starts_with("steel_attention_") {
        Some("steel_attention_")
    } else if name.starts_with("paged_steel_attention_") {
        Some("paged_steel_attention_")
    } else {
        None
    };
    if let Some(prefix) = family {
        names.extend(
            entries
                .iter()
                .filter(|entry| entry.starts_with(prefix))
                .cloned(),
        );
    }
    if name.starts_with("mlx_sdpa_vector_") && !name.contains("_2pass_") {
        names.extend(
            entries
                .iter()
                .filter(|entry| entry.starts_with("mlx_sdpa_vector_") && !entry.contains("_2pass_"))
                .cloned(),
        );
    }
    names.sort();
    names.dedup();
    names
}

pub(super) fn expand(
    request: &Request,
    dispatch: &Dispatch,
    dynamic: bool,
    entries: &HashSet<String>,
) -> Vec<Request> {
    let mut expanded = Vec::new();
    let vary_shape = dynamic || dispatch.prepared_epilogue().is_none();
    for name in alternatives(request, dispatch, entries) {
        let dense = name.starts_with("steel_gemm_");
        let mut constants = request.constants.clone();
        if name == "copy_contiguous_f32x4" {
            constants.clear();
        } else if name.starts_with("copy_") && request.name == "copy_contiguous_f32x4" {
            constants = vec![(0, 0), (2, 0)];
        }
        if name == "argmax_partials_f32x4" {
            constants.retain(|(index, _)| *index != 0);
        } else if name == "argmax_partials" && !constants.iter().any(|(index, _)| *index == 0) {
            constants.push((0, 0));
        }

        if matches!(name.as_str(), "gemv" | "gemv_transposed" | "column_gemv") {
            constants.retain(|(index, _)| !matches!(*index, 200..=202));
        }
        if dense {
            for index in 200..=202 {
                if !constants.iter().any(|(existing, _)| *existing == index) {
                    constants.push((index, 0));
                }
            }
        }
        let mut sets = vec![constants];
        let mut domains = Vec::new();
        for &(index, _) in &sets[0] {
            let values = if (vary_shape && matches!(index, 200..=202))
                || (dynamic && matches!(index, 211 | 212 | 214 | 215))
            {
                vec![0, 1]
            } else if dynamic && matches!(index, 14 | 15) {
                vec![1, 2]
            } else if vary_shape && index == 225 && name.starts_with("quantized_gemv_shared") {
                vec![1, 2, 4, 8]
            } else if vary_shape && index == 225 && name == "quantized_gemv_staged" {
                (1..=8).collect()
            } else {
                continue;
            };
            domains.push((index, values));
        }
        for (index, values) in domains {
            sets = sets
                .into_iter()
                .flat_map(|constants| {
                    values.iter().map(move |&value| {
                        constants
                            .iter()
                            .map(|&(slot, old)| (slot, if slot == index { value } else { old }))
                            .collect::<Vec<_>>()
                    })
                })
                .collect();
        }
        for constants in sets {
            if let Some(generic) = generic_constants(&name, &constants) {
                expanded.push(Request {
                    name: name.clone(),
                    constants: generic,
                });
            }
            expanded.push(Request {
                name: name.clone(),
                constants,
            });
        }
    }
    expanded
}

impl MetalBackend {
    pub(crate) fn prepare_catalog_dispatch(
        &self,
        dispatch: &Dispatch,
        dynamic: bool,
    ) -> Result<(), BackendError> {
        if dispatch.bound_program().is_some() {
            return Ok(());
        }
        let mut requests = self.catalog_requests(dispatch)?;
        if matches!(
            dispatch.op(),
            Op::Sdpa { .. } | Op::PagedSdpa { .. } | Op::PagedSdpaAppend { .. }
        ) && let Some(name) =
            crate::variants::portable_name(forja_core::OperationKind::of(dispatch.op()))
            && let Ok(variant) =
                crate::variants::resolve(dispatch, name, &[forja_core::DeviceCapability::Metal4])
        {
            requests.extend(
                self.catalog_requests(
                    &dispatch
                        .clone()
                        .with_backend_data(forja_core::BackendDispatchData::new(variant)),
                )?,
            );
        }
        let mut expanded = HashSet::new();
        for request in requests {
            expanded.extend(expand(&request, dispatch, dynamic, &self.pipelines.entries));
        }
        for request in expanded {
            if !self.pipelines.entries.contains(&request.name) {
                continue;
            }
            match self.pipelines.get(&request.name, &request.constants) {
                Ok(_) | Err(BackendError::UnsupportedOperation) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(())
    }
    fn catalog_requests(&self, dispatch: &Dispatch) -> Result<Vec<Request>, BackendError> {
        let dispatches = std::slice::from_ref(dispatch);
        let objects = self
            .in_flight
            .checkout(self, self.argument_capacity(dispatches)?)?;
        let _scratch = ScratchScope::enter(objects.scratch()?)?;
        let command = self.begin_command_buffer(objects.allocator()?)?;
        let mut arguments = ArgumentWriter::new(objects.argument_buffer()?);
        let recorder = Recorder::enter();
        self.encode_dispatches(
            &command,
            objects.argument_table()?,
            &DispatchEncoding {
                dispatches,
                barriers: &[false],
            },
            None,
            ProfileTimestampGranularity::Precise,
            &mut arguments,
            None,
            None,
        )?;
        command.endCommandBuffer();
        Ok(recorder.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::{Backend, DType, Submission};
    #[test]
    fn explicit_preparation_reuses_catalog_entries() {
        let backend = MetalBackend::new().unwrap();
        let input = backend.alloc(DType::F32, &[33]).unwrap();
        let output = backend.alloc(DType::F32, &[33]).unwrap();
        let mut commands = CommandList::new();
        commands.dispatch(Op::Copy, &[&input], &output).unwrap();
        backend
            .prepare_dispatch(commands.last_dispatch().unwrap())
            .unwrap();
        let before = backend
            .pipelines
            .pipeline_budget
            .load(std::sync::atomic::Ordering::Acquire);
        backend
            .submit_retained(commands, Vec::new())
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(
            backend
                .pipelines
                .pipeline_budget
                .load(std::sync::atomic::Ordering::Acquire),
            before
        );
    }

    #[test]
    fn dynamic_reduction_widths_consume_no_pipeline_quota_on_replay() {
        use forja_core::{
            Affine, GraphLimits, GraphTemplate, ParamSpace, SymbolicLayout, TemplateTensor,
        };
        let backend = MetalBackend::new().unwrap();
        let space = ParamSpace::new(std::iter::once(1..=4097).collect()).unwrap();
        let width = Affine::parameter(0, 0, 1);
        let input = backend.alloc(DType::F32, &[4097]).unwrap();
        let weight = backend.alloc(DType::F32, &[4097]).unwrap();
        let output = backend.alloc(DType::F32, &[4097]).unwrap();
        let ones = vec![1.0_f32.to_ne_bytes(); 4097].concat();
        backend.write(&input, &ones).unwrap();
        backend.write(&weight, &ones).unwrap();
        let symbolic = |tensor: &forja_core::Tensor| {
            TemplateTensor::symbolic(
                tensor.clone(),
                SymbolicLayout::new(tensor.layout().clone(), space.clone())
                    .slice(0, 0.into(), width, 1)
                    .unwrap(),
            )
            .unwrap()
        };
        let mut graph = GraphTemplate::new(space.clone(), GraphLimits::default());
        graph
            .dispatch(
                Op::RmsNorm { eps: 1e-5 },
                &[&symbolic(&input), &symbolic(&weight)],
                &symbolic(&output),
            )
            .unwrap();
        let graph = backend.prepare_graph(graph).unwrap();
        let before = backend
            .pipelines
            .pipeline_budget
            .load(std::sync::atomic::Ordering::Acquire);
        for width in [1, 7, 33, 4097] {
            backend.replay(&graph, vec![width]).unwrap().wait().unwrap();
            let view = backend
                .view(
                    &output,
                    forja_core::ViewOp::Slice(vec![forja_core::Slice::new(0, width, 1).unwrap()]),
                )
                .unwrap();
            let bytes = backend.read(&view).unwrap();
            let expected = vec![
                (1.0_f32 / (1.0_f32 + 1e-5).sqrt()).to_ne_bytes();
                usize::try_from(width).unwrap()
            ]
            .concat();
            forja_testing::assert_outputs_agree(DType::F32, &expected, &bytes).unwrap();
        }
        assert_eq!(
            backend
                .pipelines
                .pipeline_budget
                .load(std::sync::atomic::Ordering::Acquire),
            before
        );
    }

    #[test]
    fn constant_domains_cover_every_alignment_combination() {
        let backend = forja_cpu::CpuBackend::new();
        let input = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let weight = backend.alloc(DType::F32, &[33, 7]).unwrap();
        let output = backend.alloc(DType::F32, &[7, 7]).unwrap();
        let mut commands = CommandList::new();
        commands
            .dispatch(Op::Matmul, &[&input, &weight], &output)
            .unwrap();
        let request = Request {
            name: "steel_gemm_64_32_32_2_2".into(),
            constants: vec![(0, 0), (1, 0), (2, 0), (200, 0), (201, 0), (202, 0)],
        };
        let expanded = expand(
            &request,
            commands.last_dispatch().unwrap(),
            true,
            &HashSet::new(),
        );
        for bits in 0..8 {
            assert!(expanded.iter().any(|r| {
                (200..=202)
                    .enumerate()
                    .all(|(bit, index)| r.constants.contains(&(index, (bits >> bit) & 1)))
            }));
        }
        assert_eq!(
            generic_constants("state_delta_rule", &[(227, 1), (228, 7)]),
            Some(vec![(227, 0), (228, 0)])
        );
    }
}
