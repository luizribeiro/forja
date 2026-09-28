#[allow(unused_imports)]
use forja_sdk::Tensor;

#[forja_sdk::kernel(map)]
fn parameter_type(x: &Tensor<f32>) -> forja_sdk::kernel::Elem {
    0.0
}

fn main() {}
