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
            vulkan-loader    # libvulkan.so for non-Nix binaries
          ]));
          shellHook = ''
            ${nodeHeadersHook}
            # Expose Nix vulkan-loader to non-Nix binaries (./alpha).
            # Do NOT add system lib dirs — that clobbers Nix glibc via LD_LIBRARY_PATH > RUNPATH.
            export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:''${LD_LIBRARY_PATH}}"
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
