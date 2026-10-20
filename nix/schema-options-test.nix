{ lib }:
let
  schema = builtins.fromJSON (builtins.readFile ./host.schema.json);
  generated = import ./schema-options.nix { inherit lib; } schema;
  evaluate =
    modelPath:
    lib.evalModules {
      modules = [
        {
          options.settings = lib.mkOption {
            type = lib.types.submodule { options = generated; };
            default = { };
          };
          config.settings.model.model = modelPath;
        }
      ];
    };
  valid = evaluate "/var/lib/forja/models/qwen";
  invalid = builtins.tryEval (evaluate "relative/model").config.settings.model.model;
  rootOptions = valid.options.settings.type.getSubOptions [ ];
  modelOptions = rootOptions.model.type.getSubOptions [ ];
in
assert valid.config.settings.server.listen == "127.0.0.1:8080";
assert valid.config.settings.request.default_output_tokens == 128;
assert valid.config.settings.serving.max_batch_size == "auto";
assert modelOptions.model.description == "Model directory path.";
assert !invalid.success;
true
