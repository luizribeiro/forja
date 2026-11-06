use forja_core::{Backend, Op, Slice, Tensor, ViewOp};
use std::{error::Error, num::NonZeroU32};

type Operands = (Vec<Tensor>, Vec<Tensor>);

pub(super) fn views(
    backend: &impl Backend,
    operation: Op,
    inputs: &[&Tensor],
    outputs: &[&Tensor],
    chunk: NonZeroU32,
) -> Result<Vec<Operands>, Box<dyn Error>> {
    let (Some(input), [output]) = (inputs.first(), outputs) else {
        return Err("row chunks require inputs and one output".into());
    };
    let ([rows, _], [output_rows, _]) = (input.layout().shape(), output.layout().shape()) else {
        return Err("row chunks require matrix inputs and outputs".into());
    };
    if !matches!(operation, Op::Matmul | Op::QuantMatmul { .. })
        || rows != output_rows
        || *rows == 0
    {
        return Err("row chunks require matching nonempty matrix rows".into());
    }
    let narrow = |tensor: &Tensor, start, len| -> Result<Tensor, Box<dyn Error>> {
        Ok(backend.view(
            tensor,
            ViewOp::Slice(vec![
                Slice::new(start, len, 1)?,
                Slice::new(0, tensor.layout().shape()[1], 1)?,
            ]),
        )?)
    };
    let mut result = Vec::new();
    let mut start = 0;
    while start < *rows {
        let len = chunk.get().min(rows - start);
        let mut inputs = inputs.iter().map(|t| (*t).clone()).collect::<Vec<_>>();
        inputs[0] = narrow(input, start, len)?;
        result.push((inputs, vec![narrow(output, start, len)?]));
        start = start.checked_add(len).ok_or("row chunk offset overflow")?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use forja_core::DType;

    #[test]
    fn chunks_cover_matching_input_and_output_rows() {
        let backend = forja_cpu::CpuBackend::new();
        for (rows, limit) in [(1, 8), (7, 33), (9, 8), (16, 8), (33, 8), (4097, 8)] {
            let input = backend.alloc(DType::F32, &[rows, 7]).unwrap();
            let weight = backend.alloc(DType::F32, &[7, 33]).unwrap();
            let output = backend.alloc(DType::F32, &[rows, 33]).unwrap();
            let chunks = views(
                &backend,
                Op::Matmul,
                &[&input, &weight],
                &[&output],
                NonZeroU32::new(limit).unwrap(),
            )
            .unwrap();
            assert_eq!(u32::try_from(chunks.len()).unwrap(), rows.div_ceil(limit));
            let mut covered = 0_u32;
            for (inputs, outputs) in chunks {
                let len = inputs[0].layout().shape()[0];
                assert!(len > 0 && len <= limit);
                assert_eq!(inputs[0].layout().shape(), [len, 7]);
                assert_eq!(outputs[0].layout().shape(), [len, 33]);
                assert_eq!(
                    inputs[0].layout().offset(),
                    u64::from(covered).checked_mul(7).unwrap()
                );
                assert_eq!(
                    outputs[0].layout().offset(),
                    u64::from(covered).checked_mul(33).unwrap()
                );
                covered = covered.checked_add(len).unwrap();
            }
            assert_eq!(covered, rows);
        }
    }

    #[test]
    fn empty_rows_and_rank_one_tensors_are_refused() {
        let backend = forja_cpu::CpuBackend::new();
        let weight = backend.alloc(DType::F32, &[7, 33]).unwrap();
        for (input_shape, output_shape) in [
            (&[0, 7][..], &[0, 33][..]),
            (&[7][..], &[1, 33][..]),
            (&[1, 7][..], &[33][..]),
        ] {
            let input = backend.alloc(DType::F32, input_shape).unwrap();
            let output = backend.alloc(DType::F32, output_shape).unwrap();
            assert!(
                views(
                    &backend,
                    Op::Matmul,
                    &[&input, &weight],
                    &[&output],
                    NonZeroU32::new(8).unwrap()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn chunks_preserve_strides_and_the_tail_offset() {
        let backend = forja_cpu::CpuBackend::new();
        let storage = backend.alloc(DType::F32, &[9, 14]).unwrap();
        let input = backend
            .view(
                &storage,
                ViewOp::Slice(vec![
                    Slice::new(0, 9, 1).unwrap(),
                    Slice::new(0, 7, 2).unwrap(),
                ]),
            )
            .unwrap();
        let weight = backend.alloc(DType::F32, &[7, 33]).unwrap();
        let output = backend.alloc(DType::F32, &[9, 33]).unwrap();
        let chunks = views(
            &backend,
            Op::Matmul,
            &[&input, &weight],
            &[&output],
            NonZeroU32::new(8).unwrap(),
        )
        .unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].0[0].layout().shape(), [8, 7]);
        assert_eq!(chunks[1].0[0].layout().shape(), [1, 7]);
        assert_eq!(chunks[1].0[0].layout().strides(), input.layout().strides());
        assert_eq!(chunks[1].0[0].layout().offset(), 8 * 14);
        assert_eq!(chunks[1].1[0].layout().offset(), 8 * 33);
        let short = backend.alloc(DType::F32, &[7, 33]).unwrap();
        for (op, outputs) in [
            (Op::Matmul, vec![&short]),
            (Op::Matmul, vec![&output, &output]),
            (
                Op::QuantEmbed {
                    bits: 4,
                    group_size: 64,
                },
                vec![&output],
            ),
        ] {
            assert!(
                views(
                    &backend,
                    op,
                    &[&input, &weight],
                    &outputs,
                    NonZeroU32::new(8).unwrap()
                )
                .is_err()
            );
        }
    }
}
