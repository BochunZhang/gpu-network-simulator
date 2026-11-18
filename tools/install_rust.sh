#!/bin/sh
# Install the stable Rust toolchain for the current worker account.
set -eu

export CARGO_HOME="${CARGO_HOME:-$HOME/.cargo}"
rust_bin="$CARGO_HOME/bin"
export PATH="$rust_bin:$PATH"
export RUSTUP_TOOLCHAIN=stable

if [ ! -x "$rust_bin/rustup" ] || [ ! -x "$rust_bin/rustc" ] || [ ! -x "$rust_bin/cargo" ]; then
    installer=$(mktemp)
    trap 'rm -f "$installer"' EXIT
    curl --proto '=https' --tlsv1.2 -fsSL https://sh.rustup.rs -o "$installer"
    sh "$installer" -y --profile minimal --default-toolchain stable --no-modify-path
elif ! RUSTUP_AUTO_INSTALL=0 rustc --version >/dev/null 2>&1 ||
     ! RUSTUP_AUTO_INSTALL=0 cargo --version >/dev/null 2>&1; then
    rustup toolchain install stable --profile minimal --component rustc --component cargo
fi

rustc --version
cargo --version
printf 'Rust stable is available in %s\n' "$rust_bin"
