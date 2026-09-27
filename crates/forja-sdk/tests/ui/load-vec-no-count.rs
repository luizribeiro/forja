use forja_sdk::Load;

#[derive(Load)]
struct Layer {}

#[derive(Load)]
struct Model {
    layers: Vec<Layer>,
}

fn main() {}
