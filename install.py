#!/usr/bin/env python3
import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import docker

# ------------------------------------------------------------
# Logging helpers
# ------------------------------------------------------------

def info(msg: str):
    print(f"[INFO] {msg}")

def die(msg: str):
    print(f"[ERROR] {msg}", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------
# Resolve sima-cli (alias-safe, CI-safe)
# ------------------------------------------------------------

def resolve_sima_cli() -> str:
    env_override = os.environ.get("SIMA_CLI")
    if env_override and Path(env_override).exists():
        return env_override

    for name in ("sima-cli", "sima-cli.exe"):
        path = shutil.which(name)
        if path:
            return path

    venv_path = Path.home() / ".sima-cli" / ".venv" / "bin" / "sima-cli"
    if venv_path.exists():
        return str(venv_path)

    return ""

SIMA_CLI = resolve_sima_cli()
if not SIMA_CLI:
    die(
        "sima-cli executable not found.\n"
        "Tried:\n"
        "  - $SIMA_CLI (explicit override)\n"
        "  - PATH\n"
        "  - ~/.sima-cli/.venv/bin/sima-cli\n\n"
        "Fix by running:\n"
        "  export SIMA_CLI=/full/path/to/sima-cli"
    )

if not os.access(SIMA_CLI, os.X_OK):
    die(f"sima-cli not executable: {SIMA_CLI}")

info(f"Using sima-cli at: {SIMA_CLI}")


# ------------------------------------------------------------
# 1. Detect Palette environment
# ------------------------------------------------------------

IS_PALETTE = False
SDK_VERSION = ""

sdk_release = Path("/etc/sdk-release")
if sdk_release.exists():
    try:
        text = sdk_release.read_text()
        for line in text.splitlines():
            if line.startswith("SDK Version"):
                IS_PALETTE = True
                SDK_VERSION = line.split(":", 1)[1].strip()
                info(f"Detected Palette environment (SDK Version: {SDK_VERSION})")
                break
    except Exception:
        pass


# ------------------------------------------------------------
# 2. Palette path: local install logic
# ------------------------------------------------------------

if IS_PALETTE:
    info("Running inside Palette container. Executing container-specific install logic.")

    hostname = os.uname().nodename.lower()
    info(f"Detected container hostname: {hostname}")

    # Determine container role
    if "modelsdk" in hostname:
        req_file = "requirements_modelsdk.txt"
        container_type = "Model SDK"
    elif "mpk" in hostname:
        req_file = "requirements_mpkcli.txt"
        container_type = "MPK CLI"
    else:
        die(
            "Unable to determine Palette container type from hostname.\n"
            f"Hostname: {hostname}\n"
            "Expected hostname to contain 'modelsdk' or 'mpk'."
        )

    info(f"Detected Palette container type: {container_type}")
    info(f"Using requirements file: {req_file}")

    req_path = Path(req_file)
    if not req_path.exists():
        die(f"Required dependency file not found: {req_file}")

    # Install package itself
    subprocess.run(
        [sys.executable, "-m", "pip", "install", ".", "--force-reinstall"],
        check=True,
    )

    # Install container-specific dependencies
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file],
        check=True,
    )

    # Ensure ~/.local/bin is on PATH (Linux container)
    local_bin = Path.home() / ".local" / "bin"
    path_entries = os.environ.get("PATH", "").split(os.pathsep)

    if str(local_bin) not in path_entries:
        info(f"Adding {local_bin} to PATH")
        os.environ["PATH"] = f"{local_bin}{os.pathsep}{os.environ['PATH']}"

        shell_rc = Path.home() / ".bashrc"
        export_line = 'export PATH="$HOME/.local/bin:$PATH"'

        if shell_rc.exists():
            if export_line not in shell_rc.read_text():
                shell_rc.write_text(shell_rc.read_text() + "\n" + export_line + "\n")
                info(f"PATH persisted in {shell_rc}")
        else:
            shell_rc.write_text(export_line + "\n")
            info(f"PATH persisted in {shell_rc}")

    # Ensure `pip` command exists
    if shutil.which("pip") is None:
        local_bin.mkdir(parents=True, exist_ok=True)
        pip3 = shutil.which("pip3")
        if pip3:
            target = local_bin / "pip"
            if not target.exists():
                try:
                    target.symlink_to(pip3)
                    info("Linked pip → pip3")
                except OSError:
                    shutil.copy(pip3, target)
                    info("Copied pip3 → pip")

    info("Palette container installation completed successfully")
    sys.exit(0)

# ------------------------------------------------------------
# 3. Host path: validate SDK containers
# ------------------------------------------------------------

info("Running from the host outside of the Palette SDK containers.")
info("Checking SDK containers via Docker.")

try:
    docker_client = docker.from_env()
    containers = docker_client.containers.list()
except Exception:
    die("Docker is not installed, not running, or not accessible.")

has_model_sdk = False
has_mpk = False
has_runtime = False

for c in containers:
    image = c.image.tags[0] if c.image.tags else ""
    if "modelsdk" in image:
        has_model_sdk = True
    if "mpk_cli" in image:
        has_mpk = True
    if "elxr" in image or "yocto" in image:
        has_runtime = True

if not has_model_sdk:
    die("Model SDK container not found")
if not has_mpk:
    die("MPK CLI container not found")
if not has_runtime:
    die("Neither eLxr nor Yocto runtime container found")

info("Required SDK containers detected:")
info("  - Model SDK: OK")
info("  - MPK CLI: OK")
info("  - Runtime (eLxr or Yocto): OK")


# ------------------------------------------------------------
# 4. Install tool into SDK containers
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
# Install monitor on host
# ------------------------------------------------------------

monitor_dir = Path("model_to_pipeline") / "monitor"
if not monitor_dir.exists():
    die(f"Monitor directory not found: {monitor_dir}")

info("Installing model-to-pipeline monitor app on the host")

info("Package installation completed successfully")
print()
print("Supported YOLO models:")
print("- yolov8n, yolov8m, yolov8l")
print("- yolov9t, yolov9s, yolov9m, yolov9c")
print("- yolov10n, yolov10s, yolov10m, yolov10b, yolov10x")
print("- yolo11n, yolo11s, yolo11m, yolo11l")
print()
print("Run pipeline creation (example with YOLOv9c):")
print()
print("sima-cli sdk run tool-model-to-pipeline/samples/yolov9c/run-yaml.sima")
print()
