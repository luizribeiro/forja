//! Free-form neural-network operations.

use crate::{Element, Error, Result, Tensor, graph, sys};

/// Computes grouped-query scaled dot-product attention.
///
/// # Errors
///
/// Returns an error for incompatible shapes, types, scale, or a refused dispatch.
pub fn sdpa<T: Element>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    scale: f32,
    causal: bool,
    q_start: u32,
) -> Result<Tensor<T>> {
    let [query_heads, query_len, _] = shape3(query)?;
    let [_, _, value_width] = shape3(value)?;
    let output = Tensor::<T>::empty(vec![query_heads, query_len, value_width])?;
    graph::record(
        sys::Op::Sdpa {
            scale,
            causal,
            q_start,
        },
        &[query.handle(), key.handle(), value.handle()],
        output.handle(),
    )?;
    Ok(output)
}

fn shape3<T: Element>(tensor: &Tensor<T>) -> Result<[u32; 3]> {
    tensor
        .shape()
        .try_into()
        .map_err(|_| Error::new("attention tensors must have rank three"))
}
