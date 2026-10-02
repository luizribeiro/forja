#!/usr/bin/env bash
set -euo pipefail

repository=$(cd "$(dirname "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/forja-guest-repro.XXXXXX")
trap 'rm -rf "$scratch"' EXIT

mkdir "$scratch/source"
tar -C "$repository" -cf - \
  Cargo.toml Cargo.lock \
  crates support/forja-testing support/golden-fixtures \
  engines/olmoe engines/qwen3 engines/qwen3-coder \
  support/guests support/test-guests wit \
  | tar -C "$scratch/source" -xf -

cargo build --release --locked -p test-guests \
  --manifest-path "$repository/Cargo.toml" --target-dir "$scratch/target-a"

cargo_home=${CARGO_HOME:-${HOME}/.cargo}
ln -s "$cargo_home" "$scratch/cargo-home"
CARGO_HOME="$scratch/cargo-home" cargo build --release --locked -p test-guests \
  --manifest-path "$scratch/source/Cargo.toml" --target-dir "$scratch/target-b"

out_a=$(find "$scratch/target-a/release/build" -type d -path '*/test-guests-*/out')
out_b=$(find "$scratch/target-b/release/build" -type d -path '*/test-guests-*/out')
count=0
for component in "$out_a"/*.wasm; do
  cmp "$component" "$out_b/$(basename "$component")"
  count=$((count + 1))
done
test "$count" -gt 0
printf 'matched %d guest components across source and Cargo paths\n' "$count"
