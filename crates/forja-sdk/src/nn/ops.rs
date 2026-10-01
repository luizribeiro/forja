//! Free-form neural-network operations.

use crate::{Dim, Element, Error, FloatElement, Result, Tensor, graph, sys};

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
    q_start: impl Into<Dim>,
) -> Result<Tensor<T>> {
    let [query_heads, query_len, _] = shape3(query)?;
    let [_, _, value_width] = shape3(value)?;
    let output = Tensor::<T>::empty(vec![query_heads, query_len, value_width])?;
    sdpa_into(query, key, value, &output, scale, causal, q_start)?;
    Ok(output)
}

/// Computes grouped-query scaled dot-product attention into a supplied output view.
///
/// # Errors
///
/// Returns an error for incompatible shapes, types, scale, or a refused dispatch.
pub fn sdpa_into<T: Element>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    output: &Tensor<T>,
    scale: f32,
    causal: bool,
    q_start: impl Into<Dim>,
) -> Result<()> {
    let q_start = q_start.into();
    graph::record(
        sys::Op::Sdpa {
            scale,
            causal,
            q_start: graph::affine(&q_start)?,
        },
        &[query.handle(), key.handle(), value.handle()],
        output.handle(),
    )?;
    Ok(())
}

/// Normalizes and rotates projected queries and keys while writing keys and values into cache
/// slot views.
///
/// # Errors
///
/// Returns an error for incompatible shapes, types, parameters, aliasing, or refused work.
pub fn qkv_rope_cache_into<T: FloatElement>(
    qkv: &Tensor<T>,
    norm: &Tensor<T>,
    positions: &Tensor<T>,
    outputs: [&Tensor<T>; 3],
    eps: f32,
    theta: f32,
) -> Result<()> {
    graph::record_many(
        sys::Op::QkvRopeCache { eps, theta },
        &[qkv.handle(), norm.handle(), positions.handle()],
        &[
            outputs[0].handle(),
            outputs[1].handle(),
            outputs[2].handle(),
        ],
    )?;
    Ok(())
}

fn shape3<T: Element>(tensor: &Tensor<T>) -> Result<[u32; 3]> {
    tensor
        .shape()
        .try_into()
        .map_err(|_| Error::new("attention tensors must have rank three"))
}
