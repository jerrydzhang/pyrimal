{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixgl.url = "github:guibou/nixGL";

  };

  outputs =
    { nixpkgs, nixgl, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config = {
              allowUnfree = true;
            };
            overlays = [ nixgl.overlay ];
          };
        in

        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              uv
            ];

            shellHook = ''
              unset PYTHONPATH
              uv sync --all-extras
              . .venv/bin/activate
            '';
          };

          hpc = pkgs.mkShell {
            packages = with pkgs; [
              uv
              nixgl.default
            ];

            shellHook = ''
              unset PYTHONPATH
              uv sync --all-extras --link-mode=copy
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
