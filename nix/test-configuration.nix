{ forjaPackage }:
{ pkgs, ... }:
let
  dummyModel = pkgs.runCommand "forja-dummy-model" { } ''
    mkdir -p "$out"
    touch "$out/config.json" "$out/tokenizer.json" "$out/model.safetensors"
  '';
in
{
  nixpkgs.hostPlatform = "aarch64-darwin";
  system.stateVersion = 6;
  services.forja = {
    enable = true;
    package = forjaPackage;
    settings = {
      backend.kind = "cpu";
      model = {
        engine = toString (pkgs.writeText "engine.wasm" "dummy");
        profile = toString (pkgs.writeText "profile.toml" "dummy");
        model = toString dummyModel;
        chat_format = toString (pkgs.writeText "chat-format.wasm" "dummy");
      };
    };
  };
}
