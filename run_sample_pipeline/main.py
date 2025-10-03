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
from model_to_pipeline.utils.logger.logger import step_logger
from model_to_pipeline.utils.process_util import execute_command
from run_sample_pipeline.configs.cli_config import Config
import typer


peppi_pipeline_app = typer.Typer()


@peppi_pipeline_app.command("infer", short_help="Create a pipeline for inference")
def infer(
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
    model: str = typer.Option(..., help="Model path to create and run pipeline on"),
    rtsp_src: str = typer.Option(..., help="RTSP Stream URL"),
    host_ip: str = typer.Option(..., help="IP address of the host to visualize"),
    host_port: str = typer.Option(
        default=7000, help="Port to visualize the pipeline on"
    ),
    labels_file: str = typer.Option(
        default="",
        help="Path to labels file. If not provided, default YOLO 80 classes will be used. Required for PePPi",
    ),
    pipeline_name: str = typer.Option(
        default="MyPipeline", help="Name the pipeline. Required for PePPi"
    ),
):
    """Creates a pipeline for inference. Currently supported for model types available in --decode-type option. \
    Run this command for visualization on host:\n
    gst-launch-1.0 udpsrc port=7000 ! application/x-rtp,encoding-name=H264,payload=96 ! rtph264depay ! "video/x-h264,stream-format=byte-stream,alignment=au" ! avdec_h264 ! fpsdisplaysink
    """
    config = Config(
        device=device,
        device_ip=device_ip,
        model=model,
        rtsp_src=rtsp_src,
        host_ip=host_ip,
        host_port=host_port,
        labels_file=labels_file,
        pipeline_name=pipeline_name,
    )
    from run_sample_pipeline.create_peppi_pipeline import CreatePeppi

    pipeline = CreatePeppi(config=config)
    with step_logger(
        step_name="Preparing_Pipeline", spinner_message_prefix="Inferring"
    ):
        pipeline.prepare_pipeline_dir()

    with step_logger(step_name="Inference", spinner_message_prefix="Performing"):
        logging.info("Connecting to the device")
        op, ec = execute_command(
            f"echo n | mpk device connect -t sima@{device_ip} -p edgeai"
        )
        list(map(logging.info, op.splitlines()))
        logging.info(f"Exit code: {ec}")
        pipeline.create_mpk()
        pipeline.deploy_pipeline()
