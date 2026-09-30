//! Command-line entry point for Forja engine tooling.

mod args;
mod benchmark;
mod benchmark_record;
mod benchmark_stats;
mod config_show;
mod engine;
mod generate;
mod machine_load;
mod profile;
mod provenance;
mod verify;

use std::{error::Error, process::ExitCode};

fn main() -> ExitCode {
    let command = match args::parse(std::env::args().skip(1)) {
        Ok(command) => command,
        Err(error) => error.exit(),
    };
    match run(command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(command: args::Command) -> Result<(), Box<dyn Error>> {
    if let args::Command::Config(options) = command {
        print!("{}", config_show::render(&options)?);
        return Ok(());
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?;
    match command {
        args::Command::Bench(options) => runtime.block_on(benchmark::run(&options)),
        args::Command::Config(_) => unreachable!("configuration commands return before runtime"),
        args::Command::Profile(options) => runtime.block_on(profile::run(&options)),
        args::Command::Run(options) => runtime.block_on(generate::run(&options)),
        args::Command::Verify(options) => runtime.block_on(verify::run(&options)),
    }
}
