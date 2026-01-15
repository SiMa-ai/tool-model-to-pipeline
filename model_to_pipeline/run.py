#!/usr/bin/env python3
import os
import sys
import subprocess
import signal
import shutil
from pathlib import Path
from typing import List
import psutil
import argparse


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

MONITOR_DIR = Path("model_to_pipeline/monitor").resolve()
PORT = 5000

TMP_DIR = Path(os.getenv("TEMP", "/tmp"))
PID_FILE = TMP_DIR / "model_to_pipeline_monitor.pid"
LOG_FILE = TMP_DIR / "model_to_pipeline_monitor.log"


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def resolve_sima_run_file(input_path: str) -> Path:
    p = Path(input_path)

    if p.is_file():
        return p.resolve()

    matches = list(Path.cwd().rglob(input_path))
    if not matches:
        sys.exit(f"❌ Run file not found: {input_path}")

    if len(matches) > 1:
        print(f"❌ Multiple run files found for '{input_path}':", file=sys.stderr)
        for m in matches:
            print(f"  - {m}", file=sys.stderr)
        sys.exit("Please specify a more specific path.")

    return matches[0].resolve()


def resolve_sima_cli() -> str:
    env_override = os.environ.get("SIMA_CLI")
    if env_override and Path(env_override).exists():
        return env_override

    for name in ("sima-cli", "sima-cli.exe"):
        path = shutil.which(name)
        if path:
            return path

    venv_path = Path.home() / ".sima-cli/.venv/bin/sima-cli"
    if venv_path.exists():
        return str(venv_path)

    sys.exit("❌ sima-cli not found")


# ------------------------------------------------------------
# Port / process handling (cross-platform)
# ------------------------------------------------------------

def listening_pids(port: int) -> List[int]:
    pids = set()
    for conn in psutil.net_connections(kind="tcp"):
        if (
            conn.laddr
            and conn.laddr.port == port
            and conn.status == psutil.CONN_LISTEN
            and conn.pid
        ):
            pids.add(conn.pid)
    return sorted(pids)


def stop_monitor():
    pids = listening_pids(PORT)
    if not pids:
        print("ℹ️  No process is listening on port 5000")
        return

    print("🛑 Stopping monitor process(es) listening on port 5000:")
    for pid in pids:
        try:
            proc = psutil.Process(pid)
            print(f"   - Terminating PID={pid} ({proc.name()})")
            proc.terminate()
        except psutil.NoSuchProcess:
            pass

    # Wait briefly for clean shutdown
    psutil.wait_procs(
        [psutil.Process(pid) for pid in pids if psutil.pid_exists(pid)],
        timeout=3,
    )


def start_monitor(port=5000):
    pids = listening_pids(start_monitor)
    if pids:
        print(f"⚠️  Port {port} is already in use by:")
        for pid in pids:
            try:
                proc = psutil.Process(pid)
                print(f"   PID={pid}  CMD={' '.join(proc.cmdline())}")
            except psutil.NoSuchProcess:
                pass

        confirm = input(f"❓ Do you want to stop the process(es) using port {port}? [y/N]: ")
        if confirm.lower() != "y":
            sys.exit("❌ Aborting monitor startup.")

        stop_monitor()

    # Handle stale PID file
    if PID_FILE.exists():
        try:
            old_pid = int(PID_FILE.read_text())
            if psutil.pid_exists(old_pid):
                print(f"⚠️  Monitor already running (PID={old_pid})")
                return
            PID_FILE.unlink()
        except Exception:
            PID_FILE.unlink(missing_ok=True)

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.touch(exist_ok=True)

    print("📡 Starting monitor server...")

    with LOG_FILE.open("ab") as log:
        proc = subprocess.Popen(
            [sys.executable, f"{MONITOR_DIR}/app.py",  "--port", f"{port}"],
            cwd='.',
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

    PID_FILE.write_text(str(proc.pid))
    print(f"📡 Monitor server started (PID={proc.pid})")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def write_sima_file(yaml_arg: str) -> Path:
    """
    Create a .sima file in /tmp using the YAML path exactly as provided.
    """
    sima_path = Path("/tmp") / f"{Path(yaml_arg).stem}.sima"

    sima_contents = f"""# auto-generated.sima

model
{{
        cd /home/docker/sima-cli/tool-model-to-pipeline \\
        && .model_venv/bin/sima-model-to-pipeline model-to-pipeline --config-yaml {yaml_arg}
}}

mpk
{{
        cd /home/docker/sima-cli/tool-model-to-pipeline \\
        && ~/.local/bin/sima-model-to-pipeline model-to-pipeline --config-yaml {yaml_arg}
}}

echo "✅ All done!"
"""

    sima_path.write_text(sima_contents)
    return sima_path


def main():
    parser = argparse.ArgumentParser(
        description="Run a SiMa pipeline with monitoring support."
    )
    parser.add_argument(
        "yaml",
        help="Path to run yaml input file (container-relative path)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=5000,
        help="Port for the monitoring service (default: 5000)",
    )

    args = parser.parse_args()

    yaml_arg = args.yaml  # keep exactly as passed
    sima_cli = resolve_sima_cli()
    sima_file = write_sima_file(yaml_arg)

    def cleanup(*_):
        stop_monitor()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"🚀 Starting monitoring service on port {args.port}...")
    start_monitor(port=args.port)

    print("⚙️  Running workflow:")
    print(f"    {sima_cli} sdk run {sima_file}")

    try:
        subprocess.run(
            [sima_cli, "sdk", "run", str(sima_file)],
            check=True,
        )
    finally:
        input("🔑 Press Enter to exit...")
        cleanup()
        print("✅ All done!")



if __name__ == "__main__":
    main()
