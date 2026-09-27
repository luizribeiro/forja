#![cfg(feature = "native")]

//! Native safetensors loading coverage.

use std::path::PathBuf;

use forja_sdk::{
    Load, Tensor, Weights,
    nn::{Embedding, EmbeddingConfig, Linear, LinearConfig, RmsNorm, RmsNormConfig},
};

#[derive(Clone, Copy)]
struct Config {
    vocab: u32,
    hidden: u32,
    layers: usize,
}

#[derive(Load)]
#[load(config = Config)]
struct Block {
    #[load(prefix, config = LinearConfig::new(config.hidden, config.hidden))]
    projection: Linear<f32>,
}

#[derive(Load)]
#[load(config = Config)]
struct Model {
    #[load(
        name = "token",
        prefix,
        config = EmbeddingConfig::new(config.vocab, config.hidden)
    )]
    embedding: Embedding<f32>,
    #[load(prefix, count = config.layers)]
    blocks: Vec<Block>,
    #[load(prefix, config = RmsNormConfig::new(config.hidden, 1.0e-6))]
    norm: RmsNorm<f32>,
}

#[derive(Load)]
#[load(config = Config)]
struct FlatModel {
    block: Block,
}

#[test]
fn loads_nested_modules_from_safetensors() {
    let path = weight_file();
    let weights = Weights::open(&path).unwrap();
    let config = Config {
        vocab: 3,
        hidden: 2,
        layers: 2,
    };
    let model = Model::load(&weights, &config).unwrap();

    assert_eq!(model.blocks.len(), 2);
    let tokens = Tensor::from_slice(&[2_u32], &[1]).unwrap();
    assert_eq!(
        model.embedding.forward(&tokens).unwrap().to_vec().unwrap(),
        [5.0, 6.0]
    );
    let input = Tensor::from_slice(&[1.0_f32, 2.0], &[1, 2]).unwrap();
    assert_eq!(
        model.blocks[1]
            .projection
            .forward(&input)
            .unwrap()
            .to_vec()
            .unwrap(),
        [23.0, 53.0]
    );
    assert_eq!(model.norm.forward(&input).unwrap().shape(), [1, 2]);

    let flat = FlatModel::load(&weights, &config).unwrap();
    assert_eq!(
        flat.block
            .projection
            .forward(&input)
            .unwrap()
            .to_vec()
            .unwrap(),
        [2.0, 6.0]
    );

    let error = Model::load(
        &weights,
        &Config {
            hidden: 3,
            ..config
        },
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("token.weight"));
}

fn weight_file() -> PathBuf {
    let tensors: [(&str, Vec<u32>, Vec<f32>); 5] = [
        (
            "token.weight",
            vec![3, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ),
        (
            "blocks.0.projection.weight",
            vec![2, 2],
            vec![1.0, 0.0, 0.0, 1.0],
        ),
        (
            "blocks.1.projection.weight",
            vec![2, 2],
            vec![3.0, 10.0, 7.0, 23.0],
        ),
        ("norm.weight", vec![2], vec![1.0, 1.0]),
        ("projection.weight", vec![2, 2], vec![2.0, 0.0, 0.0, 3.0]),
    ];
    let mut offset = 0_usize;
    let entries = tensors
        .iter()
        .map(|(name, shape, values)| {
            let start = offset;
            offset += values.len() * 4;
            format!(
                "\"{name}\":{{\"dtype\":\"F32\",\"shape\":{shape:?},\"data_offsets\":[{start},{offset}]}}"
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let mut header = format!("{{{entries}}}").into_bytes();
    while !(header.len() + 8).is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = u64::try_from(header.len()).unwrap().to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(
        tensors
            .iter()
            .flat_map(|(_, _, values)| values.iter().flat_map(|value| value.to_le_bytes())),
    );
    let path = std::env::temp_dir().join(format!("forja-sdk-load-{}", std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}
