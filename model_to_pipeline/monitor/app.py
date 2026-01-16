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
#
# Copyright (c) 2026 SiMa.ai

from flask import Flask, render_template, request, abort, Response, stream_with_context
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os, re, glob, json, time, socket, logging
from datetime import datetime
import logging
import argparse
import tarfile
import tempfile

# -------------------------
# Paths & Config
# -------------------------
def get_modelsdk_host_workspace() -> str:
    """
    Locate the ModelSDK container and return the host path
    mounted at /home/docker/sima-cli.

    Returns:
        str: Host-side source path (e.g. /home/jim/workspace)

    Raises:
        RuntimeError: If container or mount cannot be found
    """

    # 1. Find modelsdk container
    try:
        ps = subprocess.check_output(
            ["docker", "ps", "--format", "{{.ID}} {{.Names}}"],
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError("Failed to run docker ps") from e

    # looking for the first model SDK container instance for now, will need to expand to support multiple versions of containers
    container_id: Optional[str] = None
    for line in ps.strip().splitlines():
        cid, name = line.split(maxsplit=1)
        if "modelsdk" in name.lower():
            container_id = cid
            break

    if not container_id:
        raise RuntimeError("ModelSDK container not found (name containing 'modelsdk')")

    # 2. Inspect mounts
    try:
        mounts_json = subprocess.check_output(
            ["docker", "inspect", container_id, "--format", "{{ json .Mounts }}"],
            text=True
        )
        mounts = json.loads(mounts_json)
    except Exception as e:
        raise RuntimeError("Failed to inspect container mounts") from e

    # 3. Find desired mount
    for m in mounts:
        if m.get("Destination") == "/home/docker/sima-cli":
            source = m.get("Source")
            if source:
                return source

    raise RuntimeError(
        "Mount to /home/docker/sima-cli not found in ModelSDK container"
    )

logging.getLogger("werkzeug").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

try:
    WORKSPACE_DIR = get_modelsdk_host_workspace()
except Exception as e:
    WORKSPACE_DIR = os.path.expanduser("~/workspace")

STATE_FILE = os.path.join(WORKSPACE_DIR, "tool-model-to-pipeline", ".model-to-pipeline-state.json")
STATE_DIR = os.path.dirname(STATE_FILE) or "."

TOOL_DIR = os.path.join(WORKSPACE_DIR, 'tool-model-to-pipeline')
LOG_DIR = os.path.join(TOOL_DIR, 'logs')

# -------------------------
# Flask App
# -------------------------

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

# Shared state (authoritative)
STEP_STATES = {}

# -------------------------
# UI Steps
# -------------------------

steps = [
    {"id": 1, "name": "Specification", "icon": "spec.svg"},
    {"id": 2, "name": "Download Model", "icon": "download.svg"},
    {"id": 3, "name": "Model Surgery", "icon": "surgery.svg"},
    {"id": 4, "name": "Download Calibration Data", "icon": "calibration.svg"},
    {"id": 5, "name": "Model Compilation", "icon": "compilation.svg"},
    {"id": 6, "name": "Pipeline Generation", "icon": "pipeline.svg"},
    {"id": 7, "name": "MPK Compilation", "icon": "calibration.svg"},
]

# -------------------------
# Helpers
# -------------------------

def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def parse_yaml_as_sections(filepath):
    sections = []
    current_section = "General"
    section_rows = []
    pending_desc = None
    stack = []

    if not filepath or not os.path.exists(filepath):
        return []

    with open(filepath, "r") as f:
        for line in f:
            raw = line.rstrip("\n")

            # Section header
            if re.match(r"^\s*#\s*-{2,}.*-{2,}\s*$", raw):
                if section_rows:
                    sections.append((current_section, section_rows))
                section = re.sub(r"^\s*#\s*-{2,}\s*(.*?)\s*-{2,}\s*$", r"\1", raw)
                current_section, section_rows = section, []
                pending_desc = None
                continue

            # Comment line
            if raw.strip().startswith("#"):
                comment_text = raw.lstrip("#").strip()
                if comment_text:
                    pending_desc = comment_text
                continue

            if ":" not in raw:
                continue

            indent = len(raw) - len(raw.lstrip())
            key, value = raw.split(":", 1)
            key, value = key.strip(), value.strip()

            level = indent // 2
            if level < len(stack):
                stack = stack[:level]
            stack.append(key)

            field_name = stack[-1]

            inline_desc = None
            if "#" in value:
                value, inline_comment = value.split("#", 1)
                value = value.strip()
                inline_desc = inline_comment.lstrip("#").strip()
            else:
                value = value.strip()

            desc = inline_desc if inline_desc else pending_desc
            pending_desc = None

            section_rows.append({
                "field": field_name,
                "value": value,
                "desc": desc if desc else ""
            })

    if section_rows:
        sections.append((current_section, section_rows))

    return sections

def find_latest_log(prefix: str):
    pattern = os.path.join(LOG_DIR, f"{prefix}_*.log")
    matches = glob.glob(pattern)
    return max(matches, key=os.path.getmtime) if matches else None


def reset_state_file():
    """Reset JSON state file and in-memory state."""
    global STEP_STATES
    STEP_STATES = {}

    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump({}, f)
        logging.info("[state] reset to empty")
    except Exception as e:
        logging.warning("[state] failed to reset: %s", e)

def load_state_file():
    """Load JSON state file safely."""
    global STEP_STATES
    try:
        with open(STATE_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            STEP_STATES = data
            logging.info("[state] updated: %s", data)
    except Exception as e:
        logging.warning("[state] failed to load: %s", e)


# -------------------------
# Watchdog Handler
# -------------------------

class StateFileHandler(FileSystemEventHandler):
    def _handle(self, path):
        if os.path.abspath(path) == os.path.abspath(STATE_FILE):
            load_state_file()

    def on_modified(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_created(self, event):
        if not event.is_directory:
            self._handle(event.src_path)

    def on_moved(self, event):
        self._handle(event.dest_path)


def start_state_watcher():
    os.makedirs(STATE_DIR, exist_ok=True)

    handler = StateFileHandler()
    observer = Observer()
    observer.schedule(handler, path=STATE_DIR, recursive=False)
    observer.start()

    logging.info("[watchdog] watching %s", STATE_FILE)
    return observer


# -------------------------
# Flask Routes
# -------------------------
@app.route("/")
def index():
    yaml_file = request.args.get("input")
    yaml_sections = []

    # 1️⃣ Fallback to STATE_FILE if no input arg
    if not yaml_file and os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
            yaml_file = state.get("yaml")
        except Exception as e:
            app.logger.exception("Failed to read state file: %s", e)
            yaml_file = None

    # 2️⃣ Load YAML sections if we have a yaml file
    if yaml_file:
        try:
            yaml_sections = parse_yaml_as_sections(yaml_file) or []
        except Exception as e:
            app.logger.exception("Failed to parse YAML: %s", e)
            yaml_sections = []

    return render_template(
        "index.html",
        steps=steps,
        yaml_file=yaml_file,
        yaml_sections=yaml_sections,
    )


@app.route("/state")
def state():
    return STEP_STATES


@app.route("/logs/<prefix>")
def get_log(prefix):
    filepath = find_latest_log(prefix)
    if not filepath:
        abort(404)
    return {
        "filename": os.path.basename(filepath),
        "content": open(filepath).read(),
    }


@app.route("/logs/<prefix>/stream")
def stream_log(prefix):
    """Stream logs to the frontend using Server-Sent Events (SSE)."""
    filepath = find_latest_log(prefix)
    if not filepath:
        abort(404, f"No log found for prefix: {prefix}")

    def generate():
        # --- Send initial file contents ---
        with open(filepath, "r") as f:
            for line in f:
                yield f"data: {line.rstrip()}\n\n"

        with open(filepath, "r") as f:
            f.seek(0, os.SEEK_END)
            while STEP_STATES.get(prefix) == "started":
                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    time.sleep(1)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/model_stats")
def model_stats():
    # 1️⃣ Load state file
    if not os.path.exists(STATE_FILE):
        abort(404, "State file not found")

    try:
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    except Exception as e:
        abort(500, f"Failed to read state file: {e}")

    model_file = state.get("model_file")
    if not model_file:
        abort(404, "Model not ready yet")

    # 2️⃣ Resolve model path (absolute or relative)
    if os.path.isabs(model_file):
        model_path = model_file
    else:
        model_path = os.path.join(TOOL_DIR, model_file)

    if not os.path.exists(model_path):
        abort(404, f"Model file not found: {model_path}")

    # 3️⃣ Extract tar.gz into /tmp
    if not tarfile.is_tarfile(model_path):
        abort(400, "Model file is not a valid tar archive")

    extract_dir = tempfile.mkdtemp(prefix="model_stats_")

    try:
        with tarfile.open(model_path, "r:*") as tar:
            tar.extractall(path=extract_dir)
    except Exception as e:
        abort(500, f"Failed to extract model archive: {e}")

    # 4️⃣ Find *_mla_stats.yaml (partial match)
    stats_file = None
    for root, _, files in os.walk(extract_dir):
        for name in files:
            if "mla_stats.yaml" in name:
                stats_file = os.path.join(root, name)
                break
        if stats_file:
            break

    if not stats_file:
        abort(404, "mla_stats.yaml not found in model archive")

    # 5️⃣ Return file content
    try:
        with open(stats_file, "r") as f:
            content = f.read()
    except Exception as e:
        abort(500, f"Failed to read stats file: {e}")

    return Response(content, mimetype="text/yaml")


# -------------------------
# Main
# -------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Model-to-Pipeline Monitor")
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the monitor server on (default: 5000)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load initial state if file exists
    if os.path.exists(STATE_FILE):
        load_state_file()

    # Start watchdog
    reset_state_file()
    observer = start_state_watcher()

    try:
        ip = get_host_ip()
        print(
            f"Monitor running at "
            f"\033[1;34mhttp://{ip}:{args.port}\033[0m"
        )

        app.run(
            host="0.0.0.0",
            port=args.port,
            debug=True
        )
    finally:
        observer.stop()
        observer.join()

