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
import tempfile

STATE_FILE = os.environ.get(
    "MODEL_TO_PIPELINE_STATE_FILE",
    "/home/docker/sima-cli/model-to-pipeline-state.json",
)

def write_state(update: dict[str, str]) -> None:
    """
    Atomically update the shared state file.
    """
    state = {}

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                state = json.load(f)
        except Exception:
            state = {}

    state.update(update)
    state["ts"] = int(time.time())

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    # Atomic write: tmp → replace
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=os.path.dirname(STATE_FILE)
    ) as tmp:
        json.dump(state, tmp)
        tmp.flush()
        os.fsync(tmp.fileno())

    os.replace(tmp.name, STATE_FILE)