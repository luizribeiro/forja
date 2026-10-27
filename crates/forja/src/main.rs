//! Command-line entry point for Forja engine tooling.

mod analysis;
mod analysis_input;
mod analysis_metrics;
mod analysis_report;
mod args;
mod benchmark;
mod benchmark_record;
mod benchmark_snapshot;
mod benchmark_stats;
mod calibration;
#[cfg(any(target_os = "macos", test))]
mod calibration_identity;
mod config_show;
mod engine;
mod generate;
mod host_schema;
mod kernel_benchmark;
mod logging;
mod machine_load;
mod profile;
mod provenance;
mod resolution;
mod serve;
#[cfg(test)]
mod serve_benchmark;
mod signals;
mod supervisor;
mod verify;

use std::{error::Error, process::ExitCode};

fn main() -> ExitCode {
    match forja_build::run_rustc_wrapper() {
        Ok(true) => return ExitCode::SUCCESS,
        Ok(false) => {}
        Err(error) => {
            eprintln!("{error}");
            return ExitCode::FAILURE;
        }
    }
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
    if let args::Command::Build(options) = command {
        println!(
            "{}",
            resolution::build(&options.profile, options.models.as_deref())?.display()
        );
        return Ok(());
    }
    if let args::Command::Catalog(options) = command {
        if options.backend == args::Backend::Cpu {
            return Err("cpu exposes no selectable algorithm variants".into());
        }
        #[cfg(target_os = "macos")]
        {
            let backend = forja_metal::MetalBackend::new()?;
            let output = if options.json {
                backend.variant_catalog_json()
            } else {
                backend.variant_catalog_markdown()
            };
            print!("{output}");
            return Ok(());
        }
        #[cfg(not(target_os = "macos"))]
        return Err("metal is available only on macOS".into());
    }
    if let args::Command::Analyze(options) = command {
        return analysis::run(&options);
    }
    if let args::Command::Calibrate(options) = command {
        return calibration::run(&options);
    }
    if let args::Command::Config(options) = command {
        print!("{}", config_show::render(&options)?);
        return Ok(());
    }
    if let args::Command::ConfigSchema = command {
        print!("{}", host_schema::render()?);
        return Ok(());
    }
    if let args::Command::ConfigCheck = command {
        return Ok(());
    }
    if let args::Command::Serve(options) = command {
        let logging = logging::init(&options.config.logging)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()?;
        return runtime.block_on(serve::run(&options, logging));
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()?;
    match command {
        args::Command::Bench(options) => runtime.block_on(benchmark::run(&options)),
        args::Command::BenchKernel(options, config) => kernel_benchmark::run(&options, &config),
        args::Command::BenchSnapshot(options, config) => benchmark_snapshot::run(&options, &config),
        args::Command::Build(_) => unreachable!("build commands return before runtime"),
        args::Command::Analyze(_) => unreachable!("analysis returns before runtime"),
        args::Command::Calibrate(_) => unreachable!("calibration returns before runtime"),
        args::Command::Catalog(_) => unreachable!("catalog commands return before runtime"),
        args::Command::Config(_) => unreachable!("configuration commands return before runtime"),
        args::Command::ConfigCheck | args::Command::ConfigSchema => {
            unreachable!("configuration commands return before runtime")
        }
        args::Command::Profile(options) => runtime.block_on(profile::run(&options)),
        args::Command::Run(options) => runtime.block_on(generate::run(&options)),
        args::Command::Serve(_) => unreachable!("serve commands use the network runtime"),
        args::Command::Verify(options) => runtime.block_on(verify::run(&options)),
    }
}
