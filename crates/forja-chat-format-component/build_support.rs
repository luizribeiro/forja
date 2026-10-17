use std::{ffi::OsString, io, path::Path, path::PathBuf};

pub(crate) fn source(
    variable: &str,
    configured: Option<OsString>,
    fallback: &Path,
    test_fixtures: bool,
) -> Result<PathBuf, io::Error> {
    let path = match configured {
        Some(path) => PathBuf::from(path),
        None if test_fixtures => fallback.to_owned(),
        None => {
            return Err(io::Error::other(format!(
                "{variable} is not set; committed fixtures require the test-fixtures feature"
            )));
        }
    };
    println!("cargo::rerun-if-changed={}", path.display());
    if path.is_file() {
        Ok(path)
    } else {
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{} does not exist", path.display()),
        ))
    }
}
