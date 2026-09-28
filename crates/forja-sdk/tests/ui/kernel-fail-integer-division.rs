#[forja_sdk::kernel(map)]
fn integer_division(
    x: forja_sdk::kernel::Elem,
    ids: forja_sdk::kernel::Elem<u32>,
) -> forja_sdk::kernel::Elem {
    ids / 2
}

fn main() {}
