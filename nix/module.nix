{
  schemaFile ? ./host.schema.json,
}:
{
  config,
  lib,
  pkgs,
  ...
}:
let
  cfg = config.services.forja;
  schema = builtins.fromJSON (builtins.readFile schemaFile);
  schemaOptions = import ./schema-options.nix { inherit lib; };
  toml = pkgs.formats.toml { };
  settings = lib.filterAttrsRecursive (_: value: value != null) cfg.settings;
  uncheckedConfig = toml.generate "forja.toml" settings;
  checkedConfig = pkgs.runCommand "forja-checked.toml" { nativeBuildInputs = [ cfg.package ]; } ''
    ${lib.getExe' cfg.package "forja"} -c ${uncheckedConfig} config check --host
    cp ${uncheckedConfig} "$out"
  '';
in
{
  options.services.forja = {
    enable = lib.mkEnableOption "the Forja inference service";
    package = lib.mkOption {
      type = lib.types.package;
      description = "Forja package containing the serving executable.";
    };
    user = lib.mkOption {
      type = lib.types.str;
      default = "_forja";
      description = "Dedicated service account.";
    };
    group = lib.mkOption {
      type = lib.types.str;
      default = "_forja";
      description = "Dedicated service group.";
    };
    uid = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 492;
      description = "Service uid; 492 provides deterministic ownership and may be overridden to avoid a host allocation.";
    };
    gid = lib.mkOption {
      type = lib.types.ints.unsigned;
      default = 492;
      description = "Service gid; 492 provides deterministic ownership and may be overridden to avoid a host allocation.";
    };
    configFile = lib.mkOption {
      type = lib.types.path;
      readOnly = true;
      default = checkedConfig;
      description = "Generated and validated host configuration.";
    };
    settings = lib.mkOption {
      type = lib.types.submodule { options = schemaOptions schema; };
      default = { };
      description = "Serving settings generated from the HostConfig schema.";
    };
  };

  config = lib.mkIf cfg.enable {
    users = {
      knownGroups = [ cfg.group ];
      knownUsers = [ cfg.user ];
      groups.${cfg.group}.gid = lib.mkDefault cfg.gid;
      users.${cfg.user} = {
        uid = lib.mkDefault cfg.uid;
        gid = lib.mkDefault cfg.gid;
        home = "/var/lib/forja";
        isHidden = true;
        shell = "/usr/bin/false";
      };
    };
    system.activationScripts.postActivation.text = lib.mkAfter ''
      install -d -m 0750 -o ${cfg.user} -g ${cfg.group} /var/lib/forja
    '';
    launchd.daemons.forja.serviceConfig = {
      ProgramArguments = [
        (lib.getExe' cfg.package "forja")
        "-c"
        (toString cfg.configFile)
        "serve"
      ];
      RunAtLoad = true;
      KeepAlive = {
        SuccessfulExit = false;
      };
      UserName = cfg.user;
      GroupName = cfg.group;
      WorkingDirectory = "/var/lib/forja";
      Umask = 27;
      SoftResourceLimits = {
        Core = 0;
        NumberOfFiles = 65536;
        NumberOfProcesses = 1024;
      };
      HardResourceLimits = {
        Core = 0;
        NumberOfFiles = 65536;
        NumberOfProcesses = 1024;
      };
    };
  };
}
