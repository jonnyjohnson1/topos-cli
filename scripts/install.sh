#!/bin/bash

# Find the 'topos-cli' directory
TOPOS_CLI_DIR=$(find "$HOME" -type d -name "topos-cli" -print -quit 2>/dev/null)

if [ -z "$TOPOS_CLI_DIR" ]; then
    echo "Error: 'topos-cli' directory not found"
    exit 1
fi

# Switch to the 'topos-cli' directory
cd "$TOPOS_CLI_DIR"

# Open a new Terminal window and run nix-shell
# Force install nix on user's system: https://github.com/DeterminateSystems/nix-installer
osascript <<EOF
tell application "Terminal"
    do script "cd '$TOPOS_CLI_DIR' && \
    if ! command -v nix &> /dev/null && [ ! -d /nix ] && [ ! -f ~/.nix-profile/etc/profile.d/nix.sh ] && [ ! -d ~/.nix-profile ]; then \
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
    fi && nix-shell --run '$SHELL'"
end tell
EOF