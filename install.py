#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
from pathlib import Path

# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------

def info(msg: str):
    print(f"[INFO] {msg}")

def die(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------
# Resolve sima-cli (stdlib only)
# ------------------------------------------------------------

def resolve_sima_cli() -> str:
    env_override = os.environ.get("SIMA_CLI")
    if env_override and Path(env_override).exists():
        return env_override

    # PATH lookup (cross-platform)
    candidates = ["sima-cli.exe", "sima-cli"] if platform.system() == "Windows" else ["sima-cli"]
    for c in candidates:
        for p in os.environ.get("PATH", "").split(os.pathsep):
            candidate = Path(p) / c
            if candidate.exists() and os.access(candidate, os.X_OK):
                return str(candidate)

    venv_path = Path.home() / ".sima-cli" / ".venv" / "bin" / "sima-cli"
    if venv_path.exists():
        return str(venv_path)

    return ""


SIMA_CLI = resolve_sima_cli()
if not SIMA_CLI:
    die(
        "sima-cli executable not found.\n"
        "Tried:\n"
        "  - $SIMA_CLI\n"
        "  - PATH\n"
        "  - ~/.sima-cli/.venv/bin/sima-cli\n\n"
        "Fix by running:\n"
        "  export SIMA_CLI=/full/path/to/sima-cli"
    )

info(f"Using sima-cli at: {SIMA_CLI}")


# ------------------------------------------------------------
# Detect Palette environment (Linux containers only)
# ------------------------------------------------------------

IS_PALETTE = False
SDK_VERSION = ""

sdk_release = Path("/etc/sdk-release")
if sdk_release.exists():
    for line in sdk_release.read_text().splitlines():
        s = line.strip()

        # Match "SDK Version" line (case-insensitive)
        if s.lower().startswith("sdk version"):
            IS_PALETTE = True
            SDK_VERSION = "unknown"

            # Preferred format: "SDK Version = <value>"
            if "=" in s:
                _, rhs = s.split("=", 1)
                SDK_VERSION = rhs.strip()

            # Alternate format: "SDK Version: <value>"
            elif ":" in s:
                _, rhs = s.split(":", 1)
                SDK_VERSION = rhs.strip()

            # Fallback: "SDK Version <value>"
            else:
                parts = s.split(None, 2) 
                if len(parts) >= 3:
                    SDK_VERSION = parts[2].strip()

            if not SDK_VERSION:
                SDK_VERSION = "unknown"

            info(f"Detected Palette environment (SDK Version: {SDK_VERSION})")
            break


# ------------------------------------------------------------
# Palette container install logic
# ------------------------------------------------------------

if IS_PALETTE:
    info("Running inside Palette container.")

    hostname = os.uname().nodename.lower()
    info(f"Detected container hostname: {hostname}")

    if "modelsdk" in hostname:
        requirements = "requirements_modelsdk.txt"
        container_type = "Model SDK"
    elif "mpk" in hostname:
        requirements = "requirements_mpkcli.txt"
        container_type = "MPK CLI"
    else:
        die(
            "Unable to determine Palette container type from hostname.\n"
            f"Hostname: {hostname}"
        )

    info(f"Detected container type: {container_type}")
    info(f"Using requirements file: {requirements}")

    if not Path(requirements).exists():
        die(f"Missing dependency file: {requirements}")

    subprocess.run(
        [sys.executable, "-m", "pip", "install", ".", "--force-reinstall"],
        check=True,
    )

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", requirements],
        check=True,
    )

    # Ensure ~/.local/bin on PATH (Linux container)
    local_bin = Path.home() / ".local" / "bin"
    if str(local_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{local_bin}:{os.environ.get('PATH','')}"

        bashrc = Path.home() / ".bashrc"
        export_line = 'export PATH="$HOME/.local/bin:$PATH"'
        if bashrc.exists() and export_line not in bashrc.read_text():
            bashrc.write_text(bashrc.read_text() + "\n" + export_line + "\n")

    info("Palette container installation completed successfully")
    sys.exit(0)


# ------------------------------------------------------------
# Host environment: validate SDK containers (Docker CLI)
# ------------------------------------------------------------

info("Running from host environment.")
info("Validating SDK containers via Docker CLI.")

# Check docker exists
try:
    subprocess.run(["docker", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
except Exception:
    die("Docker is not installed or not on PATH.")

# Get running container images
try:
    result = subprocess.run(
        ["docker", "ps", "--format", "{{.Image}}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
except subprocess.CalledProcessError as e:
    die(f"Failed to query docker containers:\n{e.stderr}")

images = result.stdout.lower().splitlines()

has_model_sdk = any("modelsdk" in img for img in images)
has_mpk = any("mpk" in img for img in images)
has_runtime = any("elxr" in img or "yocto" in img for img in images)

if not has_model_sdk:
    die("Model SDK container not found")
if not has_mpk:
    die("MPK CLI container not found")
if not has_runtime:
    die("Neither eLxr nor Yocto runtime container found")

info("Required SDK containers detected:")
info("  - Model SDK: OK")
info("  - MPK CLI: OK")
info("  - Runtime: OK")


# ------------------------------------------------------------
# Install tool into SDK containers
# ------------------------------------------------------------

info("Installing tool-model-to-pipeline into SDK containers")

subprocess.run(
    [
        SIMA_CLI,
        "sdk",
        "model",
        "sima-cli",
        "install",
        "gh:sima-ai/tool-model-to-pipeline@2.0_prep",
    ],
    check=True,
)

subprocess.run(
    [
        SIMA_CLI,
        "sdk",
        "mpk",
        "sima-cli",
        "install",
        "gh:sima-ai/tool-model-to-pipeline@2.0_prep",
    ],
    check=True,
)


# ------------------------------------------------------------
# Install monitor app on host
# ------------------------------------------------------------

info("Installing Python dependencies into local virtual environment")

# ------------------------------------------------------------
# Create venv in current directory
# ------------------------------------------------------------

venv_dir = Path(".venv")
if not venv_dir.exists():
    subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    info(f"Created virtual environment at {venv_dir}")
else:
    info(f"Using existing virtual environment at {venv_dir}")

# ------------------------------------------------------------
# Resolve pip inside venv
# ------------------------------------------------------------

pip_bin = venv_dir / (
    "Scripts/pip.exe" if platform.system() == "Windows" else "bin/pip"
)

if not pip_bin.exists():
    die(f"pip not found in virtual environment: {pip_bin}")

# ------------------------------------------------------------
# Install requirements.txt from current directory
# ------------------------------------------------------------

req_file = Path("requirements.txt")
if not req_file.exists():
    die("requirements.txt not found in current directory")

subprocess.run(
    [str(pip_bin), "install", "-r", str(req_file)],
    check=True,
)

info("Installation completed successfully into host environment")
print()
print("Supported YOLO models:")
print("- yolov8n, yolov8m, yolov8l")
print("- yolov9t, yolov9s, yolov9m, yolov9c")
print("- yolov10n, yolov10s, yolov10m, yolov10b, yolov10x")
print("- yolo11n, yolo11s, yolo11m, yolo11l")
print()
print("Run pipeline creation (example):")
print("sima-cli sdk run tool-model-to-pipeline/samples/yolov9c/run-yaml.sima")
print()
