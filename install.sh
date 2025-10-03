#!/usr/bin/env bash
set -e

# Use pip3 explicitly
pip3 install .

# Ensure ~/.local/bin is on PATH
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo "Adding $HOME/.local/bin to PATH..."
    export PATH="$HOME/.local/bin:$PATH"

    # Persist change in shell config
    shell_rc="$HOME/.bashrc"
    if [[ -n "$ZSH_VERSION" ]]; then
        shell_rc="$HOME/.zshrc"
    fi

    if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$shell_rc"; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$shell_rc"
        echo "PATH updated in $shell_rc"
    fi
fi

# Link pip → pip3 if pip is missing
if ! command -v pip >/dev/null 2>&1; then
    ln -s "$(command -v pip3)" "$HOME/.local/bin/pip"
    echo "Linked pip to pip3 in $HOME/.local/bin"
else
    echo "pip already exists, skipping symlink"
fi
