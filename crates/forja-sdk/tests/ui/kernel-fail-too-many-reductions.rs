#[forja_sdk::kernel(row)]
fn too_many_reductions(x: forja_sdk::kernel::Row) -> forja_sdk::kernel::Row {
    x.row_sum() + x.row_max() + x.row_min() + x.row_mean() + x.row_sum()
}

fn main() {}
