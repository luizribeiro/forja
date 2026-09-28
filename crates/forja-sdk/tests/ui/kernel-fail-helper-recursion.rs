#[forja_sdk::kernel(helper)]
fn recursive(x: f32) -> f32 {
    recursive(x)
}

fn main() {}
