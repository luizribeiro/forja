#[forja_sdk::kernel(row)]
fn reduction_name(x: forja_sdk::kernel::Row) -> forja_sdk::kernel::Row {
    x.sum()
}

fn main() {}
