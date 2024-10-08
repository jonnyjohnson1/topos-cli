{ pkgs ? import <nixpkgs> {} }:

let
  pythonPackages = pkgs.python3Packages;
in
  pythonPackages.callPackage ./default.nix {
    inherit (pythonPackages) buildPythonPackage pythonOlder fetchPypi flit-core pytestCheckHook;
    inherit (pkgs) lib;
  }
