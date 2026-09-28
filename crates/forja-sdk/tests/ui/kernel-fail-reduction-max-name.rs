#[forja_sdk::kernel(row)]
fn reduction_max_name(x: forja_sdk::kernel::Row) -> forja_sdk::kernel::Row {
    x.max()
}

fn main() {}
