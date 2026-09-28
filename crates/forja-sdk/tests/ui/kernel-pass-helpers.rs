#[forja_sdk::kernel(helper)]
fn square(x: f32) -> f32 {
    x * x
}

#[forja_sdk::kernel(helper)]
fn squared_sum(x: f32, y: f32) -> f32 {
    square(x) + square(y)
}

#[forja_sdk::kernel(map)]
fn map_helper(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    squared_sum(x, x)
}

fn main() {}
