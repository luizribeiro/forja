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
        cargoHook =
          {
            name,
            text,
            cargoToolchain ? toolchain,
            runtimeInputs ? [ ],
            files ? cargoFiles,
          }:
          {
            enable = true;
            entry = "${
              pkgs.writeShellApplication {
                inherit name text;
                runtimeInputs = [ cargoToolchain ] ++ runtimeInputs;
              }
            }/bin/${name}";
            inherit files;
            pass_filenames = false;
          };
        cargoHooks = {
          rustfmt =
            (cargoHook {
              name = "rustfmt-hook";
              text = "cargo fmt --all -- --check";
            })
            // {
              enable = false;
            };
        };
        offlineHooks = {
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
          ]
          ++ gitHooks.enabledPackages;
          inherit (gitHooks) shellHook;
        };
      }
    );
}
