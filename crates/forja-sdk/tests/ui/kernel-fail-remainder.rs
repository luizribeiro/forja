#[forja_sdk::kernel(map)]
fn remainder(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x % 1.0
}

fn main() {}
