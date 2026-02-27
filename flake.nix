{
  description = "Alpha — TypeScript GPU training system";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in {
      devShells = forAllSystems ({ pkgs }: let
        basePkgs = with pkgs; [ nodejs_22 gcc gnumake ];
        nodeHeadersHook = ''
          export NODE_HEADERS="$(dirname "$(dirname "$(which node)")")/include/node"
        '';
      in {
        default = pkgs.mkShell {
          name = "alpha-dev";
          packages = basePkgs ++ (with pkgs; [ bun ]);
          shellHook = ''
            ${nodeHeadersHook}
            echo "alpha dev shell (node $(node --version))"
          '';
        };

        train = pkgs.mkShell {
          name = "alpha-train";
          packages = basePkgs ++ (pkgs.lib.optionals pkgs.stdenv.isLinux (with pkgs; [
            xorg.xorgserver  # Xvfb
            vulkan-tools     # vulkaninfo
          ]));
          shellHook = ''
            ${nodeHeadersHook}
            # Do NOT set LD_LIBRARY_PATH — ./alpha uses the system linker and finds
            # system libvulkan.so natively via /etc/ld.so.cache. Adding Nix libs to
            # LD_LIBRARY_PATH causes linker incompatibilities between Nix and system glibc.
            # Headless Vulkan needs a display
            if ! pgrep -x Xvfb >/dev/null 2>&1; then
              Xvfb :99 -screen 0 1024x768x24 &>/dev/null &
              echo "Started Xvfb on :99"
            fi
            export DISPLAY=:99
            echo "alpha train shell (node $(node --version), DISPLAY=$DISPLAY)"
          '';
        };
      });
    };
}
