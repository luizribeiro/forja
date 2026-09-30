use std::{error::Error, path::Path, process::Command};

use serde::Serialize;

const RECORDED_CPU_PERCENT: f64 = 1.0;
const SIGNIFICANT_CPU_PERCENT: f64 = 5.0;

#[derive(Debug, Serialize)]
pub(crate) struct Snapshot {
    load_average: [f64; 3],
    gpu_device_percent: Option<f64>,
    busiest_processes: Vec<Process>,
}

#[derive(Debug, Serialize)]
struct Process {
    pid: u32,
    cpu_percent: f64,
    command: String,
}

pub(crate) fn capture(stage: &str) -> Result<Snapshot, Box<dyn Error>> {
    let load_average = parse_load_average(&command_output("sysctl", &["-n", "vm.loadavg"])?)?;
    let processes = command_output("ps", &["-axo", "pid=,pcpu=,comm="])?;
    let gpu = command_output(
        "ioreg",
        &["-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
    )
    .ok()
    .and_then(|output| parse_gpu_percent(&output));
    let snapshot = Snapshot {
        load_average,
        gpu_device_percent: gpu,
        busiest_processes: parse_processes(&processes, std::process::id()),
    };
    snapshot.warn(stage);
    Ok(snapshot)
}

impl Snapshot {
    fn warn(&self, stage: &str) {
        for process in self
            .busiest_processes
            .iter()
            .filter(|process| process.cpu_percent >= SIGNIFICANT_CPU_PERCENT)
        {
            eprintln!(
                "WARNING: BENCHMARK ENVIRONMENT BUSY {stage}: {} (pid {}) is using {:.1}% CPU",
                process.command, process.pid, process.cpu_percent
            );
        }
    }
}

fn command_output(program: &str, arguments: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program).args(arguments).output()?;
    if !output.status.success() {
        return Err(format!("{program} failed with {}", output.status).into());
    }
    Ok(String::from_utf8(output.stdout)?)
}

fn parse_load_average(output: &str) -> Result<[f64; 3], Box<dyn Error>> {
    let values = output
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .split_whitespace()
        .map(str::parse)
        .collect::<Result<Vec<_>, _>>()?;
    values
        .try_into()
        .map_err(|_| "sysctl returned an invalid load average".into())
}

fn parse_processes(output: &str, excluded_pid: u32) -> Vec<Process> {
    let mut processes = output
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid = fields.next()?.parse::<u32>().ok()?;
            let cpu_percent = fields.next()?.parse::<f64>().ok()?;
            let command = fields.collect::<Vec<_>>().join(" ");
            let command = Path::new(&command)
                .file_name()
                .and_then(|name| name.to_str())
                .map_or(command.as_str(), |name| name)
                .to_owned();
            (pid != excluded_pid && cpu_percent >= RECORDED_CPU_PERCENT).then_some(Process {
                pid,
                cpu_percent,
                command,
            })
        })
        .collect::<Vec<_>>();
    processes.sort_by(|left, right| right.cpu_percent.total_cmp(&left.cpu_percent));
    processes.truncate(16);
    processes
}

fn parse_gpu_percent(output: &str) -> Option<f64> {
    let (_, suffix) = output.split_once("\"Device Utilization %\"=")?;
    let value = suffix
        .chars()
        .take_while(|character| character.is_ascii_digit() || *character == '.')
        .collect::<String>();
    value.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_machine_samples() {
        let load = parse_load_average("{ 1.25 2.50 3.75 }\n").unwrap();
        assert!(
            load.iter()
                .zip([1.25, 2.5, 3.75])
                .all(|(actual, expected)| (actual - expected).abs() < f64::EPSILON)
        );
        assert_eq!(
            parse_gpu_percent("{\"Device Utilization %\"=77,\"Other\"=0}"),
            Some(77.0)
        );
        let processes = parse_processes(
            "10 2.0 /usr/bin/quiet\n11 8.5 /opt/bin/busy\n12 9.0 self\n",
            12,
        );
        assert_eq!(processes.len(), 2);
        assert_eq!(processes[0].command, "busy");
        assert!((processes[0].cpu_percent - 8.5).abs() < f64::EPSILON);
    }
}
