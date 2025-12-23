#!/usr/bin/env bash
set -euo pipefail

############################################
# Logging helpers
############################################

info() {
    echo "[INFO] $*"
}

die() {
    echo "[ERROR] $*" >&2
    exit 1
}

############################################
# Resolve sima-cli (alias-safe, CI-safe)
############################################

resolve_sima_cli() {
    # 1. Explicit override (must be a real executable)
    if [[ -n "${SIMA_CLI:-}" ]] && [[ -x "$SIMA_CLI" ]]; then
        echo "$SIMA_CLI"
        return
    fi

    # 2. PATH lookup (ignores aliases)
    if command -v sima-cli >/dev/null 2>&1; then
        command -v sima-cli
        return
    fi

    # 3. Known venv location (your setup)
    local venv_path="$HOME/.sima-cli/.venv/bin/sima-cli"
    if [[ -x "$venv_path" ]]; then
        echo "$venv_path"
        return
    fi

    return 1
}

SIMA_CLI="$(resolve_sima_cli || true)"

if [[ -z "$SIMA_CLI" ]]; then
    die "sima-cli executable not found.
Tried:
  - \$SIMA_CLI (explicit override)
  - PATH
  - $HOME/.sima-cli/.venv/bin/sima-cli

Fix by running:
  export SIMA_CLI=/full/path/to/sima-cli"
fi

info "Using sima-cli at: $SIMA_CLI"

if [[ ! -x "$SIMA_CLI" ]]; then
    die "sima-cli not found or not executable at: $SIMA_CLI
Set SIMA_CLI=/path/to/sima-cli and retry."
fi

info "Using sima-cli at: $SIMA_CLI"

############################################
# 1. Detect Palette environment
############################################

IS_PALETTE=false
SDK_VERSION=""

if [[ -f /etc/sdk-release ]]; then
    if grep -q "^SDK Version" /etc/sdk-release; then
        IS_PALETTE=true
        SDK_VERSION="$(grep '^SDK Version' /etc/sdk-release | cut -d':' -f2- | xargs)"
        info "Detected Palette environment (SDK Version: $SDK_VERSION)"
    fi
fi

############################################
# 2. Palette path: local install logic
############################################

if [[ "$IS_PALETTE" == "true" ]]; then
    info "Running inside Palette. Executing local install logic."

    pip3 install . --force-reinstall

    # Ensure ~/.local/bin is on PATH
    if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        info "Adding $HOME/.local/bin to PATH"
        export PATH="$HOME/.local/bin:$PATH"

        shell_rc="$HOME/.bashrc"
        if [[ -n "${ZSH_VERSION:-}" ]]; then
            shell_rc="$HOME/.zshrc"
        fi

        if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$shell_rc" 2>/dev/null; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$shell_rc"
            info "PATH persisted in $shell_rc"
        fi
    fi

    # Link pip → pip3 if needed
    if ! command -v pip >/dev/null 2>&1; then
        mkdir -p "$HOME/.local/bin"
        ln -sf "$(command -v pip3)" "$HOME/.local/bin/pip"
        info "Linked pip → pip3"
    else
        info "pip already exists"
    fi

    info "Palette installation completed successfully"
    exit 0
fi

############################################
# 3. Host path: validate SDK containers
############################################

info "Running from the host outside of the Palette SDK containers."
info "Checking SDK containers via Docker."

command -v docker >/dev/null 2>&1 || die "docker is not installed or not on PATH"

DOCKER_IMAGES="$(docker ps --format '{{.Image}}')"

has_model_sdk=false
has_mpk=false
has_runtime=false

while read -r image; do
    [[ "$image" == *modelsdk* ]] && has_model_sdk=true
    [[ "$image" == *mpk_cli* ]] && has_mpk=true
    [[ "$image" == *elxr* ]] || [[ "$image" == *yocto* ]] && has_runtime=true
done <<< "$DOCKER_IMAGES"

[[ "$has_model_sdk" == "true" ]] || die "Model SDK container not found"
[[ "$has_mpk" == "true" ]] || die "MPK CLI container not found"
[[ "$has_runtime" == "true" ]] || die "Neither eLxr nor Yocto runtime container found"

info "Required SDK containers detected:"
info "  - Model SDK: OK"
info "  - MPK CLI: OK"
info "  - Runtime (eLxr or Yocto): OK"

############################################
# 4. Install tool into SDK containers
############################################

info "Installing tool-model-to-pipeline into SDK containers"

"$SIMA_CLI" sdk model \
    "sima-cli install gh:sima-ai/tool-model-to-pipeline@2.0_prep"

"$SIMA_CLI" sdk mpk \
    "sima-cli install gh:sima-ai/tool-model-to-pipeline@2.0_prep"


# Install monitor on the host side

cd model_to_pipeline/monitor
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
info "Installed model-to-pipeline monitor app on the host"

info "Package installation completed successfully"
echo
echo -e "\033[1mSupported YOLO models:\033[0m"
echo "- yolov8n, yolov8m, yolov8l"
echo "- yolov9t, yolov9s, yolov9m, yolov9c"
echo "- yolov10n, yolov10s, yolov10m, yolov10b, yolov10x"
echo "- yolo11n, yolo11s, yolo11m, yolo11l"
echo

echo -e "\033[1mRun pipeline creation (example with YOLOv9c):\033[0m"
echo
echo -e "\033[32msima-cli sdk run tool-model-to-pipeline/samples/yolov9c/run-yaml.sima\033[0m"
echo