#[forja_sdk::kernel(row)]
fn reduction_name_u32(
    x: forja_sdk::kernel::Row,
    ids: forja_sdk::kernel::Row<u32>,
) -> forja_sdk::kernel::Row {
    x + ids.sum() as f32
}

fn main() {}
