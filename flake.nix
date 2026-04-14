{
  description = "Python 3.12 development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        # Use a consistent set of libraries
        runtimelibs = with pkgs; [
          stdenv.cc.cc.lib
          zlib
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [ 
            pkgs.python312 
            pkgs.uv
            pkgs.gcc 
          ];

          shellHook = ''
            # Use a helper to avoid polluting the global namespace too aggressively
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath runtimelibs}:$LD_LIBRARY_PATH"
            
            if [ ! -d ".venv" ]; then
              uv sync
            fi
            source .venv/bin/activate
          '';
        };
      });
}

