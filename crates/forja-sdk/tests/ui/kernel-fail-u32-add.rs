#[forja_sdk::kernel(map)]
fn u32_add(
    x: forja_sdk::kernel::Elem,
    ids: forja_sdk::kernel::Elem<u32>,
) -> forja_sdk::kernel::Elem {
    x + (ids + 1) as f32
}

fn main() {}
