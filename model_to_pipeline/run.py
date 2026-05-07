#!/usr/bin/env python3

# Copyright (c) 2026 SiMa.ai
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import subprocess
import signal
import shutil
import tempfile
from pathlib import Path
from typing import List
import psutil
import argparse
import webbrowser
import socket
import yaml

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

MONITOR_DIR = Path("model_to_pipeline/monitor").resolve()
PORT = 5000

TMP_DIR = Path(tempfile.gettempdir())
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
def get_host_ip() -> str:
    """
    Best-effort way to get a reachable host IP.
    Falls back to localhost.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return "127.0.0.1"


def try_open_browser(port: int):
    """
    Try to open the monitor URL in a browser if available.
    Safe for headless systems.
    """
    url = f"http://{get_host_ip()}:{port}"

    # Common headless indicators
    if os.environ.get("DISPLAY") is None and os.name != "nt":
        print(f"🖥️  No DISPLAY detected, skipping browser launch. If you have another machine that can reach this host, open the URL {url} manually.")
        return

    try:
        if webbrowser.open(url, new=2):
            print(f"🌐 Opened browser at {url}")
        else:
            print(f"⚠️  Browser detected but failed to open {url}")
    except webbrowser.Error:
        print(f"⚠️  No usable browser found, if you have another machine that can reach this host, open the URL {url} manually.")

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

def start_monitor(port=5000, log_dir=None):
    print(f"log_dir: {log_dir}")
    pids = listening_pids(port)
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

    cmd = [sys.executable, str(MONITOR_DIR / "app.py"), "--port", f"{port}"]
    if log_dir is not None:
        cmd += ["--log_dir", str(log_dir)]

    with LOG_FILE.open("ab") as log:
        proc = subprocess.Popen(
            cmd,
            cwd='.',
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )

    PID_FILE.write_text(str(proc.pid))
    print(f"📡 Monitor server started (PID={proc.pid})")
    try_open_browser(port)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def write_sima_file(yaml_arg: str, project_path: str) -> Path:
    """
    Create a .sima file in the system temp dir using the YAML path exactly as provided.
    """
    sima_path = TMP_DIR / f"{Path(yaml_arg).stem}.sima"

    # The .sima script is executed inside a Linux container by sima-cli, so all
    # embedded paths must use POSIX separators regardless of the host OS.
    yaml_arg_posix = Path(yaml_arg).as_posix()
    project_path_posix = Path(project_path).as_posix()

    sima_contents = f"""# auto-generated.sima

model
{{
        mkdir -p {project_path_posix} \\
        && cd {project_path_posix}    \\
        && /home/docker/sima-cli/tool-model-to-pipeline/.model_venv/bin/sima-model-to-pipeline model-to-pipeline --config-yaml /home/docker/sima-cli/tool-model-to-pipeline/{yaml_arg_posix}
}}

mpk
{{
        mkdir -p {project_path_posix} \\
        && cd {project_path_posix}    \\
        && ~/.local/bin/sima-model-to-pipeline model-to-pipeline --config-yaml /home/docker/sima-cli/tool-model-to-pipeline/{yaml_arg_posix}
}}

echo "✅ All done!"
"""

    sima_path.write_text(sima_contents, encoding="utf-8")
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

    yaml_arg = args.yaml

    with open(yaml_arg, "r") as f:
        yaml_contents = yaml.safe_load(f)
        project_path = yaml_contents.get("project_path")
        if not project_path:
            project_path = os.getcwd()

    sima_cli = resolve_sima_cli()
    sima_file = write_sima_file(yaml_arg, project_path)

    # The monitor server resolves log paths via the modelsdk container's bind
    # mount, which Docker Desktop reports as a WSL-internal path that Windows
    # Python cannot read. Skip it on Windows; the SDK output still streams to
    # the terminal.
    monitor_enabled = os.name != "nt"

    def cleanup(*_):
        if monitor_enabled:
            stop_monitor()

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    if monitor_enabled:
        print(f"🚀 Starting monitoring service on port {args.port}...")
        start_monitor(port=args.port, log_dir=f"{project_path}/logs")
    else:
        print("ℹ️  Monitor server is disabled on Windows; tool output will appear in this terminal.")

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
