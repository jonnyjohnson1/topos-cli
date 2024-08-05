{ pkgs ? import <nixpkgs> {} }:

let
  python = pkgs.python3;
  

in pkgs.mkShell {
  buildInputs = [
    pkgs.just
    pkgs.poetry
  ];

  shellHook = ''
    just build
    topos set --spacy trf
    topos run
  '';
}