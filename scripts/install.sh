#!/bin/bash
# Open a new Terminal window and run nix-shell
# Force install nix on user's system: https://github.com/DeterminateSystems/nix-installer
osascript <<EOF
tell application "Terminal"
    do script "if ! command -v nix &> /dev/null && [ ! -d /nix ] && [ ! -f ~/.nix-profile/etc/profile.d/nix.sh ] && [ ! -d ~/.nix-profile ]; then \
        echo 'Nix is not installed. Installing Nix...'; \
        curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install --no-confirm; \
        if [ -f ~/.nix-profile/etc/profile.d/nix.sh ]; then \
            source ~/.nix-profile/etc/profile.d/nix.sh; \
        else \
            echo 'Error: Nix profile script not found. Please check the installation.'; \
            exit 1; \
        fi; \
        echo 'Nix installation complete.'; \
    else \
        echo 'Nix is already installed.'; \
    fi && nix run github:jonnyjohnson1/topos-cli/v0.2.8 --extra-experimental-features nix-command --extra-experimental-features flakes --show-trace"
end tell
EOF