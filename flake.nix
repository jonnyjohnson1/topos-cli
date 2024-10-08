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
