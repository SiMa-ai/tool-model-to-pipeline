# Copyright (c) 2025 SiMa.ai
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
# Copyright (c) 2025 SiMa.ai

from flask import Flask, render_template, request, abort, Response, stream_with_context
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import os, re, glob, json, time, socket, logging
from datetime import datetime
import logging
import argparse

# -------------------------
# Paths & Config
# -------------------------
logging.getLogger("werkzeug").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

WORKSPACE_DIR = os.path.expanduser("~/workspace")
STATE_FILE = os.path.join(WORKSPACE_DIR, "model-to-pipeline-state.json")
STATE_DIR = os.path.dirname(STATE_FILE) or "."

LOG_DIR = os.path.join(WORKSPACE_DIR, 'model-to-pipeline', 'logs')

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

