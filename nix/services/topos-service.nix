# process-compose topos service
{ config, lib, pkgs, topos, ... }:
{
  options = {
    services.topos = {
      enable = lib.mkEnableOption "Enable topos service";
      package = lib.mkOption {
        type = lib.types.package;
        default = topos;
        description = "The topos package to use";
      };
      args = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        default = [];
        description = "Additional arguments to pass to topos";
      };
    };
  };
  config =
    let
      cfg = config.services.topos;
    in
    lib.mkIf cfg.enable {
      settings.processes.topos = {
        command = "${lib.getExe cfg.package} ${lib.escapeShellArgs cfg.args}";
      };
    };
}
