//! Process-level serving signal behavior.

#![cfg(unix)]

use std::{
    error::Error,
    fs,
    io::{Read, Write},
    net::{SocketAddr, TcpListener, TcpStream},
    path::PathBuf,
    process::{Child, Command, Output, Stdio},
    thread,
    time::{Duration, Instant},
};

use forja_config::{Profile, embed_profile, single_weights_sha256};
use forja_testing::temporary_directory;
use tokenizers::{Tokenizer, decoders::byte_level::ByteLevel, models::bpe::BPE};

struct Server {
    child: Option<Child>,
    address: SocketAddr,
    root: PathBuf,
}

impl Server {
    fn start(name: &str) -> Result<Self, Box<dyn Error>> {
        let root = temporary_directory(name)?;
        let model = root.join("model");
        fs::create_dir(&model)?;
        let weights = model.join("model.safetensors");
        let mut header = br#"{"dummy":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.to_vec();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut weight_bytes = u64::try_from(header.len())?.to_le_bytes().to_vec();
        weight_bytes.extend_from_slice(&header);
        weight_bytes.extend_from_slice(&0_f32.to_le_bytes());
        fs::write(&weights, &weight_bytes)?;
        let mut tokenizer = Tokenizer::new(BPE::default());
        tokenizer.with_decoder(Some(ByteLevel::default()));
        tokenizer
            .save(model.join("tokenizer.json"), false)
            .map_err(|error| error.to_string())?;

        let hash = single_weights_sha256(
            &weights,
            u64::try_from(weight_bytes.len()).map_err(|_| "weight length overflowed")?,
        )?;
        let profile: Profile = toml::from_str(&format!(
            r#"schema-version = 1
family = "smoke"

[model]
id = "smoke"
revision = "0000000000000000000000000000000000000000"
weights-sha256 = "{hash}"

[target]
component = "wasm32-wasip2"
backend = "cpu"
device-family = "generic"

[build]
features = []

[numerics]
activations = "f32"
weights = "f32"
kv-cache = "f32"

[workload]
kind = "causal-lm"
batch = 1
max-context = 32
prefill-chunk = 16

[serving]
paged-cache = true
"#
        ))?;
        let engine = root.join("engine.wasm");
        fs::write(
            &engine,
            embed_profile(&fs::read(test_guests::engine_smoke())?, &profile)?,
        )?;
        let profile_path = root.join("profile.toml");
        fs::write(&profile_path, profile.canonical_toml()?)?;
        let chat_format = root.join("chat-format.wasm");
        fs::copy(test_guests::chat_format_smoke(), &chat_format)?;
        let address = free_address()?;
        let metrics = free_address()?;
        let config = root.join("host.toml");
        fs::write(
            &config,
            format!(
                "[server]\nlisten='{address}'\nmodel_drain_deadline='1s'\n\
                 [model]\nengine='{}'\nprofile='{}'\nmodel='{}'\nchat_format='{}'\n\
                 [backend]\nkind='cpu'\n\
                 [metrics]\nlisten='{metrics}'\n",
                engine.display(),
                profile_path.display(),
                model.display(),
                chat_format.display(),
            ),
        )?;
        let child = Command::new(env!("CARGO_BIN_EXE_forja"))
            .args(["--isolated", "-c"])
            .arg(&config)
            .arg("serve")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut server = Self {
            child: Some(child),
            address,
            root,
        };
        server.wait_ready()?;
        Ok(server)
    }

    fn signal(&self, signal: &str) -> Result<(), Box<dyn Error>> {
        let pid = self.child.as_ref().ok_or("server already waited")?.id();
        if Command::new("kill")
            .args([signal, &pid.to_string()])
            .status()?
            .success()
        {
            Ok(())
        } else {
            Err(format!("cannot deliver {signal} to server {pid}").into())
        }
    }

    fn wait(mut self) -> Result<Output, Box<dyn Error>> {
        let output = self
            .child
            .take()
            .ok_or("server already waited")?
            .wait_with_output()?;
        fs::remove_dir_all(&self.root)?;
        if !output.status.success() {
            return Err(
                format!("serve failed: {}", String::from_utf8_lossy(&output.stderr)).into(),
            );
        }
        Ok(output)
    }

    fn wait_ready(&mut self) -> Result<(), Box<dyn Error>> {
        let started = Instant::now();
        loop {
            if let Some(status) = self.child.as_mut().ok_or("server missing")?.try_wait()? {
                let mut stderr = String::new();
                if let Some(mut stream) = self.child.as_mut().and_then(|child| child.stderr.take())
                {
                    stream.read_to_string(&mut stderr)?;
                }
                return Err(format!("serve exited before readiness ({status}): {stderr}").into());
            }
            if model_endpoint_is_ready(self.address) {
                return Ok(());
            }
            if started.elapsed() > Duration::from_mins(45) {
                return Err("serve did not become ready within 45 minutes".into());
            }
            thread::sleep(Duration::from_millis(25));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(child) = &mut self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn free_address() -> Result<SocketAddr, Box<dyn Error>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    Ok(listener.local_addr()?)
}

fn model_endpoint_is_ready(address: SocketAddr) -> bool {
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(100)) else {
        return false;
    };
    let _ = stream.set_read_timeout(Some(Duration::from_millis(100)));
    if stream
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        .is_err()
    {
        return false;
    }
    let mut response = String::new();
    stream.read_to_string(&mut response).is_ok() && response.starts_with("HTTP/1.1 200")
}

fn log_count(output: &Output, event: &str) -> usize {
    String::from_utf8_lossy(&output.stdout)
        .matches(event)
        .count()
        + String::from_utf8_lossy(&output.stderr)
            .matches(event)
            .count()
}

#[test]
fn sighups_coalesce_in_a_real_server() -> Result<(), Box<dyn Error>> {
    let server = Server::start("serve-sighup-coalesce")?;
    server.signal("-STOP")?;
    server.signal("-HUP")?;
    server.signal("-HUP")?;
    server.signal("-HUP")?;
    server.signal("-CONT")?;
    thread::sleep(Duration::from_millis(250));
    server.signal("-TERM")?;
    let output = server.wait()?;
    assert_eq!(log_count(&output, "config.reloaded"), 1);
    Ok(())
}

#[test]
fn sigterm_wins_over_sighup_in_a_real_server() -> Result<(), Box<dyn Error>> {
    let server = Server::start("serve-sigterm")?;
    server.signal("-STOP")?;
    server.signal("-HUP")?;
    server.signal("-TERM")?;
    server.signal("-CONT")?;
    let output = server.wait()?;
    assert_eq!(log_count(&output, "config.reloaded"), 0);
    Ok(())
}

#[test]
fn sigint_drains_a_real_server() -> Result<(), Box<dyn Error>> {
    let server = Server::start("serve-sigint")?;
    server.signal("-INT")?;
    let output = server.wait()?;
    assert_eq!(log_count(&output, "model.drain_finished"), 1);
    Ok(())
}
