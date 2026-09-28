#[forja_sdk::kernel(map)]
fn f32_max(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    x.max(0.0)
}

fn main() {}
