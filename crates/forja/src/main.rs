//! Command-line entry point for Forja engine tooling.

mod args;

use std::process::ExitCode;

fn main() -> ExitCode {
    match args::parse(std::env::args().skip(1)) {
        Ok(args::Command::Verify(options)) => {
            let _ = options;
            eprintln!("verification support is unavailable");
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("{error}\n{}", args::USAGE);
            ExitCode::FAILURE
        }
    }
}
