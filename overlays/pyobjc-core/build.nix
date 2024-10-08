{ pkgs ? import <nixpkgs> {} }:

let
  pythonPackages = pkgs.python3Packages;
in
  pythonPackages.callPackage ./default.nix {
    inherit (pythonPackages) buildPythonPackage setuptools;
    inherit (pkgs) lib stdenv fetchPypi xcodebuild cctools darwin;
  }
