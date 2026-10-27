use crate::{analysis_input::Profile, analysis_report::report};
use serde_json::Value;
use std::{error::Error, fs, path::PathBuf};

#[derive(Debug, PartialEq, clap::Args)]
pub(crate) struct Options {
    /// Costed JSON produced by forja profile --json.
    #[arg(long)]
    profile: PathBuf,
    /// Matching calibration JSON; omission measures/reuses the current device cache.
    #[arg(long)]
    calibration: Option<PathBuf>,
    /// Device calibration cache directory.
    #[arg(long, conflicts_with = "calibration")]
    cache: Option<PathBuf>,
    /// Fraction of measured resource ceilings used for the target.
    #[arg(long, default_value = "0.8", value_parser = utilization)]
    target: f64,
    /// Save the complete machine-readable report.
    #[arg(long)]
    json: Option<PathBuf>,
}

fn utilization(value: &str) -> Result<f64, String> {
    let value: f64 = value.parse().map_err(|_| "target must be a number")?;
    if value.is_finite() && value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err("target must be in (0, 1]".into())
    }
}

pub(crate) fn run(options: &Options) -> Result<(), Box<dyn Error>> {
    let profile: Profile = serde_json::from_slice(&fs::read(&options.profile)?)?;
    profile.validate()?;
    let calibration: Value = if let Some(path) = &options.calibration {
        serde_json::from_slice(&fs::read(path)?)?
    } else {
        crate::calibration::load_or_measure(&crate::calibration::Options {
            cache: options.cache.clone(),
            force: false,
            reps: 9,
            warmups: 3,
            json: None,
        })?
    };
    crate::calibration::validate(&calibration, &profile.device_identity)?;
    let mut report = report(&profile, &calibration, options.target)?;
    report["profile_sha256"] = golden_fixtures::sha256_file(&options.profile)?.into();
    if let Some(path) = &options.json {
        fs::write(path, serde_json::to_vec_pretty(&report)?)?;
    }
    print!("{}", crate::analysis_report::ranking::human(&report));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn targets_refuse_nonfinite_or_unattainable_fractions() {
        for value in ["0", "-1", "1.1", "NaN", "inf"] {
            assert!(utilization(value).is_err());
        }
        assert!((utilization("0.8").unwrap() - 0.8).abs() < f64::EPSILON);
    }
}
