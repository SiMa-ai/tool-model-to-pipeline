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

from flask import Flask, render_template, request, abort
import os, re, glob
from threading import Thread
from werkzeug.serving import make_server
import logging, sys
import time
from flask import Response, stream_with_context, abort
import socket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

steps = [
    {"id": 1, "name": "Specification", "icon": "spec.svg"},
    {"id": 2, "name": "Download Model", "icon": "download.svg"},
    {"id": 3, "name": "Model Surgery", "icon": "surgery.svg"},
    {"id": 4, "name": "Download Calibration Data", "icon": "calibration.svg"},
    {"id": 5, "name": "Model Compilation", "icon": "compilation.svg"},
    {"id": 6, "name": "Pipeline Generation", "icon": "pipeline.svg"},
    {"id": 7, "name": "MPK Compilation", "icon": "calibration.svg"},
]

def get_host_ip():
    """Get the primary local IP address (not 127.0.0.1)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't need to be reachable
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
    pattern = os.path.join('logs', f"{prefix}_*.log")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)

@app.route("/logs/<prefix>")
def get_log(prefix):
    filepath = find_latest_log(prefix)
    if not filepath:
        abort(404, f"No log found for prefix: {prefix}")

    with open(filepath, "r") as f:
        content = f.read()

    filename = os.path.basename(filepath)
    return {"filename": filename, "content": content}

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

        # --- Check state: only stream live if still active ---
        states = {}
        if hasattr(app, "server_thread"):
            states = app.server_thread.get_state()

        step_state = states.get(prefix)
        if step_state != "started":
            # Step is done → stop streaming
            return

        # --- Tail mode: stream new lines until state is not active ---
        with open(filepath, "r") as f:
            f.seek(0, os.SEEK_END)
            while True:
                # Re-check state each loop
                states = {}
                if hasattr(app, "server_thread"):
                    states = app.server_thread.get_state()

                step_state = states.get(prefix)
                if step_state != "started":
                    break

                line = f.readline()
                if line:
                    yield f"data: {line.rstrip()}\n\n"
                else:
                    # Send a raw heartbeat/progress tick (not parsed as event)
                    yield ".\n\n"
                    time.sleep(1)

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/state")
def state():
    if not hasattr(app, "server_thread"):
        return {}
    return app.server_thread.get_state()

@app.route("/")
def index():
    yaml_file = request.args.get("input", None)
    yaml_sections = parse_yaml_as_sections(yaml_file) if yaml_file else []
    return render_template(
        "index.html",
        steps=steps,
        yaml_file=yaml_file,
        yaml_sections=yaml_sections
    )

# --- Threaded Server Wrapper ---
class ServerThread(Thread):
    def __init__(self, app, host="0.0.0.0", port=5000, config_yaml=None):
        Thread.__init__(self)
        self.srv = make_server(host, port, app)
        self.ctx = app.app_context()
        self.ctx.push()
        self.daemon = True
        app.server_thread = self

        # Track state of steps
        self.step_states = {}

        ip = get_host_ip()
        url = f"http://{ip}:{port}?input={config_yaml}"
        print(f"Monitor Server running at \033[1;34m{url}\033[0m")

    def update_state(self, step_name: str, state) -> None:
        """Update the state of a step (started, success, fail, etc.)."""
        self.step_states[step_name] = state
        logging.info(f"[Monitor] Step {step_name} → {state}")

    def get_state(self):
        return self.step_states

    def run(self):
        print(f"Starting Flask server on port {self.srv.port} ...")
        try:
            self.srv.serve_forever()
        except Exception as e:
            logging.exception("Exception in Flask server thread: %s", e)

    def shutdown(self):
        print("Shutting down Flask server...")
        self.srv.shutdown()


if __name__ == "__main__":
    import argparse, time

    parser = argparse.ArgumentParser()
    parser.add_argument("--threaded", action="store_true", help="Run Flask in a background thread")
    args = parser.parse_args()

    if args.threaded:
        # Run using ServerThread
        server = ServerThread(app, port=5000)
        server.start()
        print("Flask server running in a thread... press Ctrl+C to stop.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            server.shutdown()
            print("Server stopped.")
    else:
        # Run standalone
        app.run(debug=True, host="0.0.0.0", port=5000)
