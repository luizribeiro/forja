#[forja_sdk::kernel(helper)]
fn square(x: f32) -> f32 {
    x * x
}

#[forja_sdk::kernel(helper)]
fn squared_sum(x: f32, y: f32) -> f32 {
    square(x) + square(y)
}

#[forja_sdk::kernel(row, helper)]
fn normalize(x: f32) -> f32 {
    x * squared_sum(x, x).row_mean().rsqrt()
}

#[forja_sdk::kernel(map)]
fn map_helper(x: forja_sdk::kernel::Elem) -> forja_sdk::kernel::Elem {
    squared_sum(x, x)
}

#[forja_sdk::kernel(row)]
fn row_helper(x: forja_sdk::kernel::Row) -> forja_sdk::kernel::Row {
    normalize(x)
}

fn main() {}
