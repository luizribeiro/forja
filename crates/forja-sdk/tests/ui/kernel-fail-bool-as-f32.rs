#[forja_sdk::kernel(map)]
fn bool_as_f32(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    (x > 0.0) as f32
}

fn main() {}
