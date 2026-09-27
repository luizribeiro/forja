use forja_sdk::{Load, Result, Weights};

struct Config {
    layers: usize,
}

struct Leaf;

impl Load<Config> for Leaf {
    fn load(_weights: &Weights<'_>, _config: &Config) -> Result<Self> {
        Ok(Self)
    }
}

#[derive(Load)]
#[load(config = Config)]
struct Block {
    #[load(prefix)]
    projection: Leaf,
}

#[derive(Load)]
#[load(config = Config)]
struct Model {
    #[load(name = "layers", prefix, count = config.layers)]
    blocks: Vec<Block>,
}

fn main() {}
