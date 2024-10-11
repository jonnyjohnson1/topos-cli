{
  description = "topos";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.flake-utils.follows = "flake-utils";
    };
  };

  outputs = { self, nixpkgs, flake-utils, poetry2nix }:
    flake-utils.lib.eachSystem ["aarch64-darwin"] (system:
      let
        pkgs = import nixpkgs {
            inherit system;
            overlays = [
            poetry2nix.overlays.default
            (final: prev: {
                myapp = final.callPackage myapp { };
                pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
                  (python-final: python-prev: {
                    pystray = python-final.callPackage ./overlays/pystray/default.nix { };
                  })
                ];
            })
            ];
        };

        # see https://github.com/nix-community/poetry2nix/tree/master#api for more functions and examples.
        myapp = { poetry2nix, lib }: poetry2nix.mkPoetryApplication {
          projectDir = self;
          preferWheels = true;
          overrides = poetry2nix.overrides.withDefaults (final: super:
            lib.mapAttrs
              (attr: systems: super.${attr}.overridePythonAttrs
                (old: {
                  nativeBuildInputs = (old.nativeBuildInputs or [ ]) ++ map (a: final.${a}) systems;
                }))
              {
                # https://github.com/nix-community/poetry2nix/blob/master/docs/edgecases.md#modulenotfounderror-no-module-named-packagename
                package = [ "setuptools" ];
              }
          );
        };

        configFile = pkgs.copyPathToStore ./config.yaml;
        yq = pkgs.yq-go;

        # Note: This only loads the settings from the repos config file
        #        if one is not already set in the user's .config directory.
        toposSetupHook = ''
          export TOPOS_CONFIG_PATH="$HOME/.config/topos/config.yaml"
          mkdir -p "$(dirname "$TOPOS_CONFIG_PATH")"
          if [ ! -f "$TOPOS_CONFIG_PATH" ]; then
            echo "Creating new config file at $TOPOS_CONFIG_PATH"
            echo "# Topos Configuration" > "$TOPOS_CONFIG_PATH"
            ${yq}/bin/yq eval ${configFile} | while IFS= read -r line; do
              echo "$line" >> "$TOPOS_CONFIG_PATH"
            done
            echo "Config file created at $TOPOS_CONFIG_PATH"
          else
            echo "Config file already exists at $TOPOS_CONFIG_PATH"
          fi
        '';
      in
      {
        packages = {
            default = pkgs.myapp;
            topos = pkgs.writeShellScriptBin "topos" ''
            export PATH="${pkgs.myapp}/bin:$PATH"
            ${pkgs.myapp}/bin/topos run
            '';
        };

        devShells = {
          # Shell for app dependencies.
          #
          #     nix develop
          #
          # Use this shell for developing your app.
          default = pkgs.mkShell {
            inputsFrom = [  pkgs.myapp ];

            shellHook = ''
              export PATH="${pkgs.myapp}/bin:$PATH"
              ${toposSetupHook}
            '';
          };

          # Shell for topos
          #
          #     nix develop .#topos
          #
          # Use this shell running topos
          topos = pkgs.mkShell {
            inputsFrom = [  pkgs.myapp ];

            shellHook = ''
              export PATH="${pkgs.myapp}/bin:$PATH"
              ${toposSetupHook}
              topos run
            '';
          };

          # Shell for poetry.
          #
          #     nix develop .#poetry
          #
          # Use this shell for changes to pyproject.toml and poetry.lock.
          poetry = pkgs.mkShell {
            packages = [ pkgs.poetry ];
          };
        };
        legacyPackages = pkgs;
      }
    );
}
