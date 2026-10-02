//! Model weight identity checks.

use std::fs;

use forja_config::{sharded_weights_sha256, single_weights_sha256};
use forja_testing::temporary_directory;

#[test]
fn hashes_one_file_by_its_bytes() {
    let root = temporary_directory("single-weights-hash").unwrap();
    let path = root.join("model.safetensors");
    fs::write(&path, b"weights").unwrap();
    assert_eq!(
        single_weights_sha256(&path, 7).unwrap(),
        "sha256:9a129038d9a00aed0cf6a7ea059ca50a813449061ab87848cf1a13eafdf33b2c"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn shard_hash_covers_sorted_names_lengths_and_bytes() {
    let root = temporary_directory("sharded-weights-hash").unwrap();
    fs::write(root.join("a.safetensors"), b"ab").unwrap();
    fs::write(root.join("bc.safetensors"), b"c").unwrap();
    let a = root.join("a.safetensors");
    let bc = root.join("bc.safetensors");
    let first = sharded_weights_sha256(&[
        ("a.safetensors", a.as_path(), 2),
        ("bc.safetensors", bc.as_path(), 1),
    ])
    .unwrap();

    fs::write(root.join("a.safetensors"), b"a").unwrap();
    fs::write(root.join("bc.safetensors"), b"bc").unwrap();
    assert_ne!(
        first,
        sharded_weights_sha256(&[
            ("a.safetensors", a.as_path(), 1),
            ("bc.safetensors", bc.as_path(), 2),
        ])
        .unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn rejects_empty_and_escaping_shard_names() {
    let root = temporary_directory("invalid-shard-hash").unwrap();
    assert_eq!(
        sharded_weights_sha256(&[]).unwrap_err().kind(),
        std::io::ErrorKind::InvalidInput
    );
    assert_eq!(
        sharded_weights_sha256(&[("../model.safetensors", root.as_path(), 0)])
            .unwrap_err()
            .kind(),
        std::io::ErrorKind::InvalidInput
    );
    fs::remove_dir_all(root).unwrap();
}
