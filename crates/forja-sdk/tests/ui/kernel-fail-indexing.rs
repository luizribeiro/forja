#[forja_sdk::kernel(map)]
fn indexing(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x[0]
}

fn main() {}
