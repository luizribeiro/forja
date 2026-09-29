use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

use crate::{ConfigError, DevConfig, Layer, Origin, Schema, layer::validate};

const MODELS_ENV: &str = "FORJA_MODELS";

/// Returns the XDG user configuration path when a home directory is available.
#[must_use]
pub fn user_config_path() -> Option<PathBuf> {
    user_config_path_from(
        env::var_os("XDG_CONFIG_HOME").map(PathBuf::from),
        env::var_os("HOME").map(PathBuf::from),
    )
}

fn user_config_path_from(xdg: Option<PathBuf>, home: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(path) = xdg.filter(|path| path.is_absolute()) {
        return Some(path.join("forja/forja.toml"));
    }
    home.map(|path| path.join(".config/forja/forja.toml"))
}

/// Builds the environment and optional user-file layers for development commands.
///
/// # Errors
///
/// Returns an error when `FORJA_MODELS` is not Unicode or the user file cannot be read or does not
/// match the development schema.
pub fn dev_layers(isolated: bool) -> Result<Vec<Layer>, ConfigError> {
    dev_layers_from(
        env::var_os(MODELS_ENV).map(PathBuf::from),
        (!isolated).then(user_config_path).flatten(),
    )
}

/// Reads and validates one explicit configuration file.
///
/// # Errors
///
/// Returns an error naming the file and line when it cannot be read, parsed, or validated.
pub fn file_layer<C: Schema>(path: &Path) -> Result<Layer, ConfigError> {
    read_layer::<C>(path, false)?
        .ok_or_else(|| ConfigError::new(format!("{}: file does not exist", path.display())))
}

fn dev_layers_from(
    models: Option<PathBuf>,
    user_path: Option<PathBuf>,
) -> Result<Vec<Layer>, ConfigError> {
    let mut layers = Vec::new();
    if let Some(models) = models {
        let models = models
            .to_str()
            .ok_or_else(|| ConfigError::new(format!("{MODELS_ENV}: path is not valid UTF-8")))?;
        let source = format!(
            "[paths]\nmodels = {}\n",
            toml::Value::String(models.to_owned())
        );
        let table = toml::from_str(&source)
            .map_err(|error: toml::de::Error| ConfigError::new(error.message()))?;
        validate::<DevConfig>(&table)
            .map_err(|message| ConfigError::new(format!("{MODELS_ENV}: {message}")))?;
        layers.push(Layer::new(Origin::Env(MODELS_ENV), table));
    }
    if let Some(path) = user_path
        && let Some(layer) = read_layer::<DevConfig>(&path, true)?
    {
        layers.push(layer);
    }
    Ok(layers)
}

fn read_layer<C: Schema>(path: &Path, user: bool) -> Result<Option<Layer>, ConfigError> {
    let source = match fs::read_to_string(path) {
        Ok(source) => source,
        Err(error) if user && error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(ConfigError::new(format!("{}: {error}", path.display())));
        }
    };
    let table: toml::Table =
        toml::from_str(&source).map_err(|error| source_error(path, &source, &error))?;
    if user && table.contains_key("engine") {
        let line = top_level_key_line(&source, "engine");
        return Err(ConfigError::new(format!(
            "{}:{line}: engine.* is not allowed in the user file",
            path.display()
        )));
    }
    toml::from_str::<C>(&source).map_err(|error| source_error(path, &source, &error))?;
    let origin = if user {
        Origin::UserFile(path.to_owned())
    } else {
        Origin::File(path.to_owned())
    };
    Ok(Some(Layer::new(origin, table)))
}

fn source_error(path: &Path, source: &str, error: &toml::de::Error) -> ConfigError {
    let line = error.span().map_or(1, |span| {
        source[..span.start].bytes().filter(|&b| b == b'\n').count() + 1
    });
    ConfigError::new(format!("{}:{line}: {}", path.display(), error.message()))
}

fn top_level_key_line(source: &str, key: &str) -> usize {
    source
        .lines()
        .position(|line| {
            let line = line.trim_start();
            line.starts_with(&format!("[{key}"))
                || line.starts_with(&format!("{key}."))
                || line.starts_with(&format!("{key} ="))
                || line.starts_with(&format!("\"{key}\""))
        })
        .map_or(1, |index| index + 1)
}

#[cfg(test)]
mod tests {
    use forja_testing::temporary_directory;

    use super::*;
    use crate::{KeyPath, Paths, layer};

    #[test]
    fn layers_models_before_the_user_file() {
        let root = temporary_directory("config-user").unwrap();
        let path = root.join("forja.toml");
        fs::write(&path, "[paths]\nmodels = \"/user/models\"\n").unwrap();
        let layers = dev_layers_from(Some("/env/models".into()), Some(path.clone())).unwrap();
        let layered = layer::<DevConfig>(layers).unwrap();
        assert_eq!(
            layered.config.paths,
            Paths {
                models: Some("/user/models".into()),
                scratch: "target/forja-bench".into(),
            }
        );
        assert_eq!(
            layered.origins.get(&KeyPath::new("paths.models")),
            Some(&Origin::UserFile(path))
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn user_file_rejects_engine_keys_with_a_line() {
        let root = temporary_directory("config-engine").unwrap();
        let path = root.join("forja.toml");
        fs::write(
            &path,
            "[limits]\ntensor_rank = 7\n[engine]\nprofile = \"fast\"\n",
        )
        .unwrap();
        let error = dev_layers_from(None, Some(path.clone())).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "{}:3: engine.* is not allowed in the user file",
                path.display()
            )
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn file_schema_errors_name_the_line() {
        let root = temporary_directory("config-line").unwrap();
        let path = root.join("bad.toml");
        fs::write(&path, "[limits]\ntensor_rank = 7\ntensor_ranks = 8\n").unwrap();
        let error = file_layer::<DevConfig>(&path).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "{}:3: unknown field `tensor_ranks`, expected one of `live_bytes`, `tensor_rank`, `tensor_elements`, `live_tensor_handles`, `live_kernels`, `live_graphs`, `read_bytes`, `guest_memory_bytes`, `table_elements`, `instances`, `dispatches_per_list`, `work_per_dispatch`, `guest_call_timeout`, `submission_timeout`, `gpu_time_budget`",
                path.display()
            )
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn resolves_the_user_config_path_from_xdg_then_home() {
        assert_eq!(
            user_config_path_from(Some("/xdg".into()), Some("/home/test".into())),
            Some("/xdg/forja/forja.toml".into())
        );
        assert_eq!(
            user_config_path_from(Some("relative".into()), Some("/home/test".into())),
            Some("/home/test/.config/forja/forja.toml".into())
        );
        assert_eq!(user_config_path_from(None, None), None);
    }
}
