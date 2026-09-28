#[forja_sdk::kernel(map)]
fn integer_literal(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x + 2
}

fn main() {}
