{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    nix-darwin = {
      url = "github:nix-darwin/nix-darwin";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    git-hooks = {
      url = "github:cachix/git-hooks.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nix-darwin,
      nixpkgs,
      flake-utils,
      git-hooks,
      rust-overlay,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
        toolchain = pkgs.rust-bin.stable."1.99.1".minimal.override {
          extensions = [
            "clippy"
            "rust-analyzer"
            "rust-src"
            "rustfmt"
          ];
          targets = [
            "wasm32-unknown-unknown"
            "wasm32-wasip2"
          ];
        };
        rustPlatform = pkgs.makeRustPlatform {
          cargo = toolchain;
          rustc = toolchain;
        };
        forjaPackage = rustPlatform.buildRustPackage {
          pname = "forja";
          version = "0.0.0";
          src = pkgs.lib.fileset.toSource {
            root = ./.;
            fileset = pkgs.lib.fileset.unions [
              ./Cargo.lock
              ./Cargo.toml
              ./chat-formats
              ./crates
              ./engines
              ./support
              ./wit
            ];
          };
          cargoLock.lockFile = ./Cargo.lock;
          cargoBuildFlags = [
            "-p"
            "forja"
            "--bin"
            "forja"
          ];
          doCheck = false;
          FORJA_BUILD_REV = self.rev or self.dirtyRev or "unknown";
        };
        cargoFiles = "(^|/)(Cargo\\.(toml|lock)|.*\\.rs)$";
        runEverythingFiles = "(Cargo\\.lock|flake\\.lock|rust-toolchain(\\.toml)?)$";
        metalFiles = "^(${runEverythingFiles}|(wit|engines|chat-formats|support/(guests|test-guests)|crates/forja-(core|host|metal|sdk[^/]*))/|.*\\.metal$)";
        guestFiles = "^(${runEverythingFiles}|(wit|engines|chat-formats|support/(guests|test-guests)|crates/forja-(chat-format|program-conversions|sdk[^/]*))/)";
        qwenFiles = "^(${runEverythingFiles}|(wit|engines|chat-formats|support/(golden-fixtures|guests|test-guests)|crates/forja-(core|host|metal|sdk[^/]*))/|.*\\.metal$)";
        pythonFiles = "^support/.*\\.py$";
        allLocalStages = [
          "pre-commit"
          "pre-push"
        ];
        cargoHook =
          {
            name,
            text,
            cargoToolchain ? toolchain,
            runtimeInputs ? [ ],
            files ? cargoFiles,
            stages ? [ "pre-commit" ],
          }:
          {
            enable = true;
            entry = "${
              pkgs.writeShellApplication {
                inherit name text;
                runtimeInputs = [ cargoToolchain ] ++ runtimeInputs;
              }
            }/bin/${name}";
            inherit files stages;
            pass_filenames = false;
          };
        cargoHooks = {
          rustfmt = cargoHook {
            name = "rustfmt-hook";
            stages = allLocalStages;
            text = ''
              cargo fmt --all -- --check
              for manifest in support/guests/Cargo.toml engines/*/Cargo.toml; do
                cargo fmt --all --manifest-path "$manifest" -- --check
              done
            '';
          };
          host-clippy = cargoHook {
            name = "host-clippy-hook";
            stages = allLocalStages;
            text = "cargo clippy --workspace --all-targets --all-features --locked -- -D warnings";
          };
          wasm-clippy = cargoHook {
            name = "wasm-clippy-hook";
            files = "^(chat-formats/|support/guests/|engines/)";
            stages = allLocalStages;
            text = ''
              for manifest in support/guests/Cargo.toml engines/*/Cargo.toml; do
                cargo clippy --manifest-path "$manifest" --workspace --all-targets --target wasm32-wasip2 --locked -- -D warnings
              done
              for manifest in chat-formats/*/Cargo.toml; do
                cargo clippy --manifest-path "$manifest" --all-targets --target wasm32-unknown-unknown --locked -- -D warnings
              done
            '';
          };
          cargo-nextest = cargoHook {
            name = "cargo-nextest-hook";
            runtimeInputs = [ pkgs.cargo-nextest ];
            text = "CARGO_INCREMENTAL=1 cargo nextest run --profile ci --release --workspace --all-features --locked --no-tests pass";
          };
          cargo-nextest-full = cargoHook {
            name = "cargo-nextest-full-hook";
            runtimeInputs = [
              pkgs.cargo-nextest
              pkgs.git
            ];
            files = "(${cargoFiles}|${metalFiles})";
            stages = [ "pre-push" ];
            text = ''
              if [[ "''${FORJA_NO_GPU:-}" == "1" ]]; then
                if [[ "''${GITHUB_ACTIONS:-}" != "true" ]]; then
                  echo "FORJA_NO_GPU is reserved for GitHub Actions" >&2
                  exit 1
                fi
              fi

              run_everything=false
              run_non_metal=false
              run_metal=false

              classify_path() {
                local path="$1"
                case "$path" in
                  Cargo.lock | flake.lock | rust-toolchain | rust-toolchain.toml)
                    run_everything=true
                    ;;
                esac
                case "$path" in
                  Cargo.toml | */Cargo.toml | *.rs)
                    run_non_metal=true
                    ;;
                esac
                case "$path" in
                  wit/* | engines/* | chat-formats/* | support/guests/* | support/test-guests/* | crates/forja-core/* | crates/forja-host/* | crates/forja-metal/* | crates/forja-sdk*/* | *.metal)
                    run_metal=true
                    ;;
                esac
              }

              if [[ -n "''${PRE_COMMIT_FROM_REF:-}" && -n "''${PRE_COMMIT_TO_REF:-}" ]]; then
                while IFS= read -r -d "" path; do
                  classify_path "$path"
                done < <(git diff --name-only -z "$PRE_COMMIT_FROM_REF" "$PRE_COMMIT_TO_REF")
              else
                run_everything=true
              fi

              if $run_everything || { $run_non_metal && $run_metal; }; then
                if [[ "''${FORJA_NO_GPU:-}" == "1" ]]; then
                  CARGO_INCREMENTAL=1 cargo nextest run --profile ci --release --workspace --all-features --locked --no-tests pass
                else
                  CARGO_INCREMENTAL=1 cargo nextest run --profile ci --release --workspace --all-features --locked --no-tests pass
                  CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=target/non-incremental cargo nextest run --release --workspace --all-features --locked --no-tests pass -E 'package(forja-metal) or test(/^native_metal::/) or test(/^kernel_ui::/) or test(/(^|::)metal_/)'
                fi
              elif $run_non_metal; then
                CARGO_INCREMENTAL=1 cargo nextest run --profile ci --release --workspace --all-features --locked --no-tests pass
              elif $run_metal && [[ "''${FORJA_NO_GPU:-}" != "1" ]]; then
                CARGO_INCREMENTAL=0 CARGO_TARGET_DIR=target/non-incremental cargo nextest run --release --workspace --all-features --locked --no-tests pass -E 'package(forja-metal) or test(/^native_metal::/) or test(/^kernel_ui::/) or test(/(^|::)metal_/)'
              fi
            '';
          };
          guest-reproducibility = cargoHook {
            name = "guest-reproducibility-hook";
            files = guestFiles;
            stages = [ "pre-push" ];
            text = "support/test-guests/check-reproducible.sh";
          };
          cargo-deny = cargoHook {
            name = "cargo-deny-hook";
            runtimeInputs = [ pkgs.cargo-deny ];
            files = "(^|/)(Cargo\\.(toml|lock)|deny\\.toml)$";
            stages = allLocalStages;
            text = ''
              for manifest in Cargo.toml engines/*/Cargo.toml; do
                cargo deny --manifest-path "$manifest" check bans licenses sources
              done
            '';
          };
          doctests =
            (cargoHook {
              name = "doctests-hook";
              text = "cargo test --doc --workspace --all-features --locked";
            })
            // {
              stages = [ "pre-push" ];
            };
          docs =
            (cargoHook {
              name = "docs-hook";
              text = ''
                RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --all-features --locked
              '';
            })
            // {
              stages = [ "pre-push" ];
            };
          qwen-weights =
            (cargoHook {
              name = "qwen-weights-hook";
              runtimeInputs = [ pkgs.cargo-nextest ];
              files = qwenFiles;
              text = ''
                if [[ "''${GITHUB_ACTIONS:-}" == "true" ]]; then
                  echo "skipping model tests: GitHub Actions runners have no model weights"
                  exit 0
                fi
                export FORJA_MODELS="''${FORJA_MODELS:-$HOME/.cache/forja/models}"
                export CARGO_INCREMENTAL=0
                export CARGO_TARGET_DIR=target/non-incremental
                cargo nextest run --release -p forja-host --test host -E 'test(qwen_weights::)' --run-ignored only
                cargo nextest run --release -p forja-sdk --features native-metal --test sdk -E 'test(qwen_moe::)' --run-ignored only
                cargo nextest run --release -p forja-host --test host -E 'test(qwen_engine::)' --run-ignored only
                cargo nextest run --release -p forja-host --test host -E 'test(qwen3_coder_engine::)' --run-ignored only
                cargo nextest run --release -p golden-fixtures --test fixtures -E 'test(qwen_fixtures::)' --run-ignored only
                cargo nextest run --release -p golden-fixtures --test fixtures -E 'test(mixture_fixtures::)' --run-ignored only
                cargo nextest run --release -p forja --bin forja --run-ignored only
              '';
            })
            // {
              stages = [ "pre-push" ];
            };
        };
        pythonHooks = {
          python-checks = {
            enable = true;
            entry = "${
              pkgs.writeShellApplication {
                name = "python-checks-hook";
                runtimeInputs = [
                  pkgs.python3
                  pkgs.python3Packages.pytest
                  pkgs.ruff
                ];
                text = ''
                  pytest -q support
                  ruff check --select E4,E7,E9,F support
                '';
              }
            }/bin/python-checks-hook";
            files = pythonFiles;
            pass_filenames = false;
            stages = [ "pre-commit" ];
          };
        };
        offlineHooks = pkgs.lib.mapAttrs (_: hook: hook // { stages = allLocalStages; }) {
          benchmark-record-size = {
            enable = true;
            entry = "${
              pkgs.writeShellApplication {
                name = "benchmark-record-size-hook";
                text = ''
                  readonly maximum_bytes=65536
                  failed=0
                  for record in "$@"; do
                    [[ -f "$record" ]] || continue
                    bytes=$(wc -c < "$record")
                    if (( bytes > maximum_bytes )); then
                      echo "$record is $bytes bytes; committed benchmark records must not exceed $maximum_bytes bytes" >&2
                      failed=1
                    fi
                  done
                  exit "$failed"
                '';
              }
            }/bin/benchmark-record-size-hook";
            files = "^bench/.*\\.json$";
            pass_filenames = true;
          };
          target-deps-entries = {
            enable = true;
            entry = "${
              pkgs.writeShellApplication {
                name = "target-deps-entries-hook";
                text = ''
                  readonly maximum_entries=20000
                  failed=0
                  shopt -s nullglob
                  for deps in target/*/deps target/*/*/deps target/tests/trybuild/*/*/deps; do
                    entries=$(find "$deps" -mindepth 1 -maxdepth 1 | wc -l)
                    if (( entries > maximum_entries )); then
                      echo "$deps holds $entries entries (limit $maximum_entries). Metal device creation scans the test executable's directory, so every Metal test slows down. Run: find target -path '*/deps/*.rcgu.o' -delete (or cargo clean --profile dev)" >&2
                      failed=1
                    fi
                  done
                  exit "$failed"
                '';
              }
            }/bin/target-deps-entries-hook";
            always_run = true;
            pass_filenames = false;
          };
          nixfmt.enable = true;
          deadnix.enable = true;
          statix.enable = true;
          taplo.enable = true;
          actionlint.enable = true;
          typos.enable = true;
          check-merge-conflicts.enable = true;
          end-of-file-fixer.enable = true;
          trim-trailing-whitespace.enable = true;
          check-yaml.enable = true;
          check-toml.enable = true;
        };
        hookDefinitions = offlineHooks // cargoHooks // pythonHooks;
        gitHooks = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = hookDefinitions;
        };
      in
      {
        packages = {
          default = forjaPackage;
          forja = forjaPackage;
        };
        checks = {
          pre-commit = git-hooks.lib.${system}.run {
            src = ./.;
            hooks = offlineHooks;
          };
        }
        // pkgs.lib.optionalAttrs (system == "aarch64-darwin") {
          forja-darwin-build = self.darwinConfigurations.forja-test.system;
          forja-module-eval =
            let
              configuration = self.darwinConfigurations.forja-test;
              configFile = configuration.config.services.forja.configFile;
              service = configuration.config.launchd.daemons.forja.serviceConfig;
              arguments = service.ProgramArguments;
            in
            assert service.UserName == "_forja";
            assert configuration.config.users.users._forja.isHidden;
            assert configuration.options.services.forja.uid.default == 492;
            assert configuration.options.services.forja.gid.default == 492;
            assert
              arguments == [
                "${configuration.config.services.forja.package}/bin/forja"
                "-c"
                "${configFile}"
                "serve"
              ];
            assert builtins.all (pkgs.lib.hasPrefix "/") [
              (builtins.elemAt arguments 0)
              (builtins.elemAt arguments 2)
            ];
            assert service.WorkingDirectory == "/var/lib/forja";
            assert service.Umask == 27;
            assert service.RunAtLoad;
            assert !service.KeepAlive.SuccessfulExit;
            assert service.SoftResourceLimits.Core == 0;
            assert service.SoftResourceLimits.NumberOfFiles == 65536;
            assert service.SoftResourceLimits.NumberOfProcesses == 1024;
            assert service.HardResourceLimits.Core == 0;
            assert service.HardResourceLimits.NumberOfFiles == 65536;
            assert service.HardResourceLimits.NumberOfProcesses == 1024;
            pkgs.runCommand "forja-module-eval" { } ''
              grep -q 'live_bytes = "auto"' ${configFile}
              grep -q 'kind = "cpu"' ${configFile}
              grep -q '^engine = "/nix/store/' ${configFile}
              grep -q '^chat_format = "/nix/store/' ${configFile}
              touch "$out"
            '';
        };

        devShells.default = pkgs.mkShell {
          FORJA_BUILD_REV = self.rev or self.dirtyRev or "unknown";
          packages = [
            toolchain
            pkgs.wasmtime
            pkgs.wasm-tools
            pkgs.cargo-nextest
            pkgs.cargo-deny
            pkgs.git-absorb
            pkgs.uv
            (pkgs.writeShellScriptBin "forja" ''
              root="''${FORJA_ROOT:-$(git rev-parse --show-toplevel)}"
              exec cargo run --release --quiet --manifest-path "$root/Cargo.toml" -p forja -- "$@"
            '')
          ]
          ++ gitHooks.enabledPackages;
          shellHook = ''
            export FORJA_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
            export FORJA_MODELS="''${FORJA_MODELS:-$HOME/.cache/forja/models}"
            ${gitHooks.shellHook}
          '';
        };
      }
    )
    // {
      darwinModules.forja = import ./nix/module.nix { };
      darwinConfigurations.forja-test = nix-darwin.lib.darwinSystem {
        modules = [
          self.darwinModules.forja
          (import ./nix/test-configuration.nix {
            forjaPackage = self.packages.aarch64-darwin.forja;
          })
        ];
      };
    };
}
