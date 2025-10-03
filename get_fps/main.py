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

import logging
import sys
import time
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box

import os
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from model_to_pipeline.utils.logger.logger import step_logger
from model_to_pipeline.utils.proto.ssh_proto import SSH

import typer


fps_getter_app = typer.Typer()


@fps_getter_app.command("get-fps")
def get_fps(
    device: str = typer.Option(
        default="davinci",
        help="Type of the board to use for compilation.",
        show_choices=True,
        case_sensitive=False,
        prompt=False,
        metavar="davinci/modalix",
    ),
    device_ip: str = typer.Option(
        ...,
        help="Provide device IP address",
        case_sensitive=False,
        prompt=False,
    ),
    models: str = typer.Option(..., help="List of model paths (comma-separated)"),
):
    """Finds the FPS number for MLA only using mla-rt"""
    ssh = SSH(host=device_ip)
    resutls = []
    device_to_args_mapper = {
        "davinci": {
            "mlart_args": " -e0 -e1 -e3 -e4 -I4dm ",
            "model_usec_regex": re.compile(r"^avg(?P<model_usec>\s+\d+\|){4}.*"),
        },
        "modalix": {
            "mlart_args": " -e1 -e2 -e6 -e7 ",
            "model_usec_regex": re.compile(r"^avg(?P<model_usec>\s+\d+\|){3}.*"),
        },
    }
    for model in models.split(','):
        start_time = time.time()
        with step_logger(
            step_name=os.path.basename(model),
            spinner_message_prefix="Finding FPS for: ",
        ):
            with TemporaryDirectory() as tmpdir:
                fps = 0
                shutil.unpack_archive(filename=model, extract_dir=tmpdir)
                model_file = [f for f in os.listdir(tmpdir) if f.endswith("elf")][0]
                logging.info(f"Pushing {model_file} to device")
                ssh.push(source=Path(tmpdir) / model_file, dest="/tmp/")
                op, ec = ssh.execute_and_get_op(
                    f"mla-rt -vs -m 1000 /tmp/{model_file} {device_to_args_mapper.get(device).get('mlart_args')}"
                )
                logging.info(op)
                logging.info(ec)
                regex = device_to_args_mapper.get("modalix").get("model_usec_regex")
                for line in op.splitlines():
                    matched = regex.match(line)
                    if matched:
                        fps = float(
                            1e6 / int(matched.group("model_usec").strip().split("|")[0])
                        )
                        logging.info(f"FPS: {fps}")
            logging.info(f"Cleaning up pushed file")
            ssh.execute_and_get_op(command=f"rm -rf /tmp/{model_file}")
        elapsed_time = time.time() - start_time
        resutls.append([os.path.basename(model), elapsed_time, fps])
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    # Summary
    console = Console()
    console.print("\n[bold underline][/bold underline]")
    # Summary table at the end
    table = Table(
        title="SiMa.ai Model MLA FPS",
        caption="Summary",
        box=box.ROUNDED,
        padding=(0, 1, 0, 1),
        show_lines=True,
        row_styles=["cyan", "magenta"],
    )
    table.add_column("Model Name", justify="center", no_wrap=True, style="bold")
    table.add_column("Elapsed Time", justify="center", style="cyan", no_wrap=True)
    table.add_column("FPS", justify="center", style="cyan", no_wrap=True)

    for model, elapsed_time, fps in resutls:
        model_name = Text(model, style="green")
        elapsed_time = Text(f"{elapsed_time:.2f} sec")
        fps = Text(f"{float(fps):.3f}")
        table.add_row(model_name, elapsed_time, fps)

    console.print(table)
    console.print("\n[bold underline][/bold underline]")

def main():
    fps_getter_app()


if __name__ == "__main__":
    main()
