use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::{Component, Path},
};

use sha2::{Digest, Sha256};

/// Hashes one unsharded safetensors file.
///
/// # Errors
///
/// Returns an error when the file cannot be opened or read.
pub fn single_weights_sha256(path: &Path, expected_length: u64) -> io::Result<String> {
    let mut hash = Sha256::new();
    hash_file(path, expected_length, &mut hash)?;
    Ok(finish(hash))
}

/// Hashes every unique shard named by a safetensors index.
///
/// Names, their byte lengths, file lengths, and file bytes enter the hash in sorted name order.
///
/// # Errors
///
/// Returns an error when no shard is referenced, a name is not a normalized relative UTF-8 path,
/// arithmetic overflows, or a shard cannot be opened or read completely.
pub fn sharded_weights_sha256(shards: &[(&str, &Path, u64)]) -> io::Result<String> {
    if shards.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "safetensors index references no shards",
        ));
    }
    let mut hash = Sha256::new();
    let mut previous = None;
    for &(name, path, expected_length) in shards {
        validate_name(name)?;
        if previous.is_some_and(|previous| previous >= name) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "safetensors shard names must be unique and sorted",
            ));
        }
        previous = Some(name);
        let name_len = u64::try_from(name.len())
            .map_err(|_| io::Error::other("shard name length overflows u64"))?;
        hash.update(name_len.to_le_bytes());
        hash.update(name.as_bytes());
        hash.update(expected_length.to_le_bytes());
        hash_file(path, expected_length, &mut hash)?;
    }
    Ok(finish(hash))
}

fn validate_name(name: &str) -> io::Result<()> {
    let path = Path::new(name);
    if name.is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid relative shard name {name:?}"),
        ));
    }
    Ok(())
}

fn hash_file(path: &Path, expected_length: u64, hash: &mut Sha256) -> io::Result<()> {
    if !path.symlink_metadata()?.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("weight source is not a regular file: {}", path.display()),
        ));
    }
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() != expected_length {
        return Err(io::Error::other(format!(
            "weight length changed before hashing: {}",
            path.display()
        )));
    }
    let limit = expected_length
        .checked_add(1)
        .ok_or_else(|| io::Error::other("weight length overflows u64"))?;
    let mut reader = BufReader::new(file.take(limit));
    let mut buffer = vec![0_u8; 64 * 1024].into_boxed_slice();
    let mut length = 0_u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            if length == expected_length {
                return Ok(());
            }
            return Err(io::Error::other(format!(
                "weight length changed while hashing: {}",
                path.display()
            )));
        }
        length = length
            .checked_add(
                u64::try_from(read).map_err(|_| io::Error::other("read length overflows u64"))?,
            )
            .ok_or_else(|| io::Error::other("weight length overflows u64"))?;
        hash.update(&buffer[..read]);
        if length > expected_length {
            return Err(io::Error::other(format!(
                "weight length changed while hashing: {}",
                path.display()
            )));
        }
    }
}

fn finish(hash: Sha256) -> String {
    format!("sha256:{:x}", hash.finalize())
}
