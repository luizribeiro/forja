//! Command-line entry point for Forja engine tooling.

mod args;
mod benchmark;
mod benchmark_stats;
mod engine;
mod generate;
mod verify;

use std::{error::Error, process::ExitCode};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let command = args::parse(std::env::args().skip(1))
        .map_err(|error| format!("{error}\n{}", args::USAGE))?;
    let runtime = tokio::runtime::Builder::new_current_thread().build()?;
    match command {
        args::Command::Bench(options) => runtime.block_on(benchmark::run(&options)),
        args::Command::Run(options) => runtime.block_on(generate::run(&options)),
        args::Command::Verify(options) => runtime.block_on(verify::run(&options)),
    }
}
