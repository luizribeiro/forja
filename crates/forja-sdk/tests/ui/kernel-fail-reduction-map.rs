#[forja_sdk::kernel(map)]
fn reduction_map(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x.row_sum()
}

fn main() {}
