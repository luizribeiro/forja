#[forja_sdk::kernel(map)]
fn output_bool(
    x: forja_sdk::kernel::Elem,
) -> (forja_sdk::kernel::Elem, forja_sdk::kernel::Elem) {
    (x, x > 0.0)
}

fn main() {}
