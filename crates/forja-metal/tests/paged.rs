//! Paged-copy CPU/Metal differentials.

#![cfg(target_os = "macos")]
#![allow(clippy::cast_precision_loss)]

use forja_core::{Backend, BackendError, CommandList, DType, Op, PagedCopyDirection, Submission};

fn encode(values: &[f32], dtype: DType) -> Vec<u8> {
    match dtype {
        DType::F32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DType::F16 | DType::BF16 | DType::I32 | DType::U32 => unreachable!(),
    }
}

fn table_bytes(entries: &[(u32, u32)]) -> Vec<u8> {
    entries
        .iter()
        .flat_map(|&(page, generation)| [page, generation])
        .flat_map(u32::to_le_bytes)
        .collect()
}

fn generation_bytes(generations: &[u32]) -> Vec<u8> {
    generations
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn run_copy<B: Backend>(backend: &B) -> Result<Vec<u8>, BackendError> {
    let source = backend.alloc(DType::F32, &[7, 3])?;
    let pool = backend.alloc(DType::F32, &[3, 4, 3])?;
    let table = backend.alloc(DType::U32, &[2, 2])?;
    let generations = backend.alloc(DType::U32, &[3])?;
    let output = backend.alloc(DType::F32, &[7, 3])?;
    let values = (0_u32..21)
        .map(|value| value as f32 / 8.0)
        .collect::<Vec<_>>();
    backend.write(&source, &encode(&values, DType::F32))?;
    backend.write(&table, &table_bytes(&[(2, 5), (0, 7)]))?;
    backend.write(&generations, &generation_bytes(&[7, 0, 5]))?;
    let mut commands = CommandList::new();
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::IntoPages,
                start: 1,
                page_tokens: 4,
            },
            &[&source, &table, &generations],
            &pool,
        )
        .map_err(|_| BackendError::InvalidInput)?;
    commands
        .dispatch(
            Op::PagedCopy {
                direction: PagedCopyDirection::OutOfPages,
                start: 1,
                page_tokens: 4,
            },
            &[&pool, &table, &generations],
            &output,
        )
        .map_err(|_| BackendError::InvalidInput)?;
    backend.submit(commands)?.wait()?;
    backend.read(&output)
}

#[test]
fn metal_paged_copy_matches_cpu_across_page_edges() {
    let cpu = forja_cpu::CpuBackend::new();
    let metal = forja_metal::MetalBackend::new().unwrap();
    assert_eq!(run_copy(&metal).unwrap(), run_copy(&cpu).unwrap());
}
