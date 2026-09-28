{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
        toolchain = pkgs.rust-bin.stable.latest.minimal.override {
          extensions = [
            "clippy"
            "rust-analyzer"
            "rust-src"
            "rustfmt"
          ];
          targets = [ "wasm32-wasip2" ];
        };
        cargoFiles = "(^|/)(Cargo\\.(toml|lock)|.*\\.rs)$";
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
              cargo fmt --all --manifest-path support/guests/Cargo.toml -- --check
            '';
          };
          host-clippy = cargoHook {
            name = "host-clippy-hook";
            stages = allLocalStages;
            text = "cargo clippy --workspace --all-targets --all-features --locked -- -D warnings";
          };
          wasm-clippy = cargoHook {
            name = "wasm-clippy-hook";
            files = "^support/guests/";
            stages = allLocalStages;
            text = "cargo clippy --manifest-path support/guests/Cargo.toml --workspace --all-targets --target wasm32-wasip2 --locked -- -D warnings";
          };
          cargo-nextest = cargoHook {
            name = "cargo-nextest-hook";
            runtimeInputs = [ pkgs.cargo-nextest ];
            text = "cargo nextest run --profile ci --release --workspace --all-features --locked --no-tests pass";
          };
          cargo-nextest-full = cargoHook {
            name = "cargo-nextest-full-hook";
            runtimeInputs = [ pkgs.cargo-nextest ];
            stages = [ "pre-push" ];
            text = ''
              if [[ "''${FORJA_NO_GPU:-}" == "1" ]]; then
                if [[ "''${GITHUB_ACTIONS:-}" != "true" ]]; then
                  echo "FORJA_NO_GPU is reserved for GitHub Actions" >&2
                  exit 1
                fi
                cargo nextest run --profile ci --release --workspace --all-features --locked --no-tests pass
              else
                cargo nextest run --release --workspace --all-features --locked --no-tests pass
              fi
            '';
          };
          cargo-deny = cargoHook {
            name = "cargo-deny-hook";
            runtimeInputs = [ pkgs.cargo-deny ];
            files = "(^|/)(Cargo\\.(toml|lock)|deny\\.toml)$";
            stages = allLocalStages;
            text = "cargo deny check bans licenses sources";
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
              text = ''
                if [[ "''${GITHUB_ACTIONS:-}" == "true" ]]; then
                  echo "skipping model tests: GitHub Actions runners have no model weights"
                  exit 0
                fi
                export FORJA_MODELS="''${FORJA_MODELS:-$HOME/.cache/forja/models}"
                cargo nextest run --release -p forja-host --test qwen_weights --run-ignored only
                cargo nextest run --release -p forja-host --test qwen_engine --run-ignored only
                cargo nextest run --release -p golden-fixtures --test qwen_fixtures --run-ignored only
                cargo nextest run --release -p forja --bin forja --run-ignored only
              '';
            })
            // {
              stages = [ "pre-push" ];
            };
        };
        offlineHooks = pkgs.lib.mapAttrs (_: hook: hook // { stages = allLocalStages; }) {
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
        hookDefinitions = offlineHooks // cargoHooks;
        gitHooks = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = hookDefinitions;
        };
      in
      {
        checks.pre-commit = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = offlineHooks;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            toolchain
            pkgs.wasmtime
            pkgs.wasm-tools
            pkgs.cargo-nextest
            pkgs.cargo-deny
            pkgs.git-absorb
            pkgs.uv
          ]
          ++ gitHooks.enabledPackages;
          shellHook = ''
            export FORJA_MODELS="''${FORJA_MODELS:-$HOME/.cache/forja/models}"
            ${gitHooks.shellHook}
          '';
        };
      }
    );
}
