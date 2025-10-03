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

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

import yaml
from model_to_pipeline.compilers.compiler_base import CompilerMeta
import typer

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
import logging
import traceback
from model_to_pipeline.utils.logger.logger import (
    step_logger,
)  # Assuming your logger setup is here
from model_to_pipeline.steps.steps_base import StepMeta
from model_to_pipeline.utils.yaml_display import build_tables_from_yaml


model_to_pipeline_app = typer.Typer()


def run_step(
    step_name: str, args: argparse.Namespace, console: Console, max_name_len: int
) -> bool:
    """Runs individual step

    Args:
        step_name (str): Name of the step being executed
        args (argparse.Namespace): All command line args
        console (Console): Console instance from rich module
        max_name_len (int): Max length of the name to display

    Raises:
        RuntimeError: If there's an excetion in running the step

    Returns:
        bool: If step execution is successful
    """
    step_class = StepMeta.registry.get(step_name)
    if not step_class:
        logging.error(f"Step '{step_name}' is not registered.")
        console.print(f":cross_mark: [bold red]Step '{step_name}' is not registered.")
        return False

    with step_logger(step_name=step_name, log_dir="logs"):
        logging.info("=" * 50)
        logging.info(f"Executing step: {step_name}")
        logging.info("=" * 50)
        try:
            result = step_class().run(args)
            if result:
                logging.info(f"Step '{step_name}' executed successfully.")
                console.print(
                    f":white_check_mark: Step [bold]{step_name:<{max_name_len}}[/] [green]completed successfully[/]"
                )
                return True
            else:
                logging.error(f"Step '{step_name}' failed to execute.")
                console.print(
                    f":cross_mark: Step [bold]{step_name:<{max_name_len}}[/] [red]failed[/]"
                )
                raise RuntimeError
        except Exception as e:
            logging.error(f"Error executing step '{step_name}': {e}")
            console.print(
                f":cross_mark: Step [bold]{step_name:<{max_name_len}}[/] [red]errored[/]"
            )
            traceback.print_exc()
            raise RuntimeError


def main(args: argparse.Namespace) -> None:
    """Main function which executes the tool

    Args:
        args (argparse.Namespace): Command line args to the tool
    """
    console = Console()
    logging.info("Starting the Sima Model to Pipeline conversion process.")

    steps = list(StepMeta.registry.items())
    max_name_len = max(len(name) for name, _ in steps)
    results = []

    with step_logger(step_name="setup", log_dir="logs"):
        logging.info(args)
        logging.info(f"Total number of steps: {len(steps)}")

    for step_name, _ in steps:
        if args.step and args.step != step_name:
            continue
        start_time = time.time()
        success = run_step(step_name, args, console, max_name_len)
        elapsed = time.time() - start_time
        results.append((step_name, success, elapsed))

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    # Summary
    console.print("\n[bold underline][/bold underline]")
    # Summary table at the end
    table = Table(
        title="SiMa.ai Model to Pipeline Summary",
        caption="Summary",
        box=box.ROUNDED,
        padding=(0, 1, 0, 1),
        show_lines=True,
        row_styles=["cyan", "magenta"],
    )
    table.add_column("Step Name", justify="center", no_wrap=True, style="bold")
    table.add_column("Elapsed Time", justify="center", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center", style="cyan", no_wrap=True)

    for step, success, elapsed in results:
        status = Text("PASS", style="green") if success else Text("FAIL", style="red")
        elapsed = Text(f"{elapsed:.2f} sec")
        table.add_row(step, status, elapsed)

    console.print(table)
    console.print("\n[bold underline][/bold underline]")


@model_to_pipeline_app.command("model-to-pipeline")
def run(
    # Model args
    model_path: Optional[str] = typer.Option(
        None, help="Path to the model file to use."
    ),
    model_name: Optional[str] = typer.Option(
        None,
        help="Name of the model.",
        show_choices=True,
        case_sensitive=False,
        prompt=False,
        metavar="yolov8/yolov9/yolov8-seg/yolov10/yolo11/yolo11-seg",
    ),
    post_surgery_model_path: Optional[str] = typer.Option(
        None, help="Path to the model file after surgery."
    ),
    model_type: Optional[str] = typer.Option(
        "object-detection",
        help="Type of the model being used",
        show_choices=True,
        case_sensitive=False,
        prompt=False,
        metavar="object-detection/image-classification/segmentation",
    ),
    input_width: Optional[int] = typer.Option(None, help="Input width of the pipeline"),
    input_height: Optional[int] = typer.Option(
        None, help="Input height of the pipeline"
    ),
    # Compilation step
    compilation_result_dir: str = typer.Option(
        "result",
        help="Model compilation result dir. Compiled model will be dumped here",
    ),
    compiler: str = typer.Option(
        list(CompilerMeta.registry.keys())[0],
        help="Name of compiler",
        case_sensitive=False,
        show_choices=True,
        prompt=False,
        metavar="/".join(list(CompilerMeta.registry.keys())),
    ),
    calibration_data_path: Optional[str] = typer.Option(
        "/home/docker/calibration_images", help="Path to the calibration dataset."
    ),
    calibration_samples_count: Optional[int] = typer.Option(
        None, help="Max number of calibration samples."
    ),
    arm_only: bool = typer.Option(
        False, help="If set, the model will be compiled for ARM architecture only."
    ),
    act_asym: bool = typer.Option(
        True,
        help="If set, the model will be compiled with asym activation quantization",
    ),
    act_per_ch: bool = typer.Option(
        False,
        help="If set, the model will be compiled with per_ch activation quantization",
    ),
    act_bf16: bool = typer.Option(
        False, help="Keep precision as bf16 for activation quantization?"
    ),
    act_nbits: int = typer.Option(
        8,
        help="Activation quantization Bit precision for compilation",
        show_choices=True,
        metavar="4/8/16",
    ),
    wt_asym: bool = typer.Option(
        False, help="If set, the model will be compiled with asym weight quantization"
    ),
    wt_per_ch: bool = typer.Option(
        True, help="If set, the model will be compiled with per_ch weight quantization"
    ),
    wt_bf16: bool = typer.Option(
        False, help="Keep precision as bf16 for weight quantization?"
    ),
    wt_nbits: int = typer.Option(
        8,
        help="Weight quantization Bit precision for compilation",
        show_choices=True,
        metavar="4/8/16",
    ),
    bias_correction: Optional[str] = typer.Option(
        "none",
        help="Bias Correction type",
        show_choices=True,
        case_sensitive=False,
        prompt=False,
        metavar="none/iterative/regular",
    ),
    ceq: bool = typer.Option(False, help="Enable channel equalization"),
    smooth_quant: bool = typer.Option(False, help="If true, smooth_quant is enabled"),
    compress: bool = typer.Option(False, help="Compress while compiling"),
    mode: Optional[str] = typer.Option(
        "sima",
        help="Requantization Mode",
        show_choices=True,
        case_sensitive=False,
        prompt=False,
        metavar="sima/tflite",
    ),
    calibration_type: Optional[str] = typer.Option(
        "mse",
        help="Type of calibration to use",
        show_choices=True,
        case_sensitive=False,
        prompt=False,
        metavar="min_max/moving_average/entropy/percentile/mse",
    ),
    batch_size: int = typer.Option(1, help="Batch size"),
    calibration_ds_extn: str = typer.Option(
        "jpg", help="Extension of the calibration dataset files."
    ),
    device_type: Optional[str] = typer.Option(
        "davinci",
        help="Type of the board to use for compilation.",
        show_choices=True,
        case_sensitive=False,
        prompt=False,
        metavar="davinci/modalix/both",
    ),
    # Pipeline creation
    input_resource: Optional[str] = typer.Option(
        None, help="Path to input image or video."
    ),
    pipeline_name: Optional[str] = typer.Option(
        None, help="Final name of the pipeline."
    ),
    config_yaml: Optional[str] = typer.Option(
        None, help="Provide configuration from config yaml file"
    ),
    # Extra
    no_box_decode: bool = typer.Option(
        False, help="Use detessdequant instead of simaaiboxdecode"
    ),
    rtsp_src: Optional[str] = typer.Option(
        None, help="RTSP Stream in case of RTSP pipeline"
    ),
    host_ip: Optional[str] = typer.Option(
        None, help="Host IP to stream back the video in case of RTSP pipeline"
    ),
    host_port: Optional[str] = typer.Option(
        None, help="Host port to stream back the video in case of RTSP pipeline"
    ),
    detection_threshold: float = typer.Option(
        0.4, help="Input detection_threshold of the pipeline"
    ),
    nms_iou_threshold: float = typer.Option(
        0.4, help="Input nms_iou_threshold of the pipeline"
    ),
    topk: int = typer.Option(25, help="Expected topk value for the pipeline"),
    labels_file: Optional[str] = typer.Option(
        None, help="Path to the labels file for the model."
    ),
    num_classes: int = typer.Option(80, help="Number of classes in the model"),
    step: Optional[str] = typer.Option(
        None, help="Mention step name to execute. Other steps will be skipped"
    ),
) -> None:
    """Runs the tool to convert the model into a working pipeline"""
    args = argparse.Namespace()
    args.model_path = model_path
    args.model_name = model_name
    args.post_surgery_model_path = post_surgery_model_path
    args.model_type = model_type
    args.input_width = input_width
    args.input_height = input_height
    args.compilation_result_dir = compilation_result_dir
    args.compiler = compiler
    args.calibration_data_path = calibration_data_path
    args.calibration_samples_count = calibration_samples_count
    args.arm_only = arm_only
    # Activation quantization
    args.act_asym = act_asym
    args.act_per_ch = act_per_ch
    args.act_bf16 = act_bf16
    args.act_nbits = act_nbits
    # Weight quantization
    args.wt_asym = wt_asym
    args.wt_per_ch = wt_per_ch
    args.wt_bf16 = wt_bf16
    args.wt_nbits = wt_nbits
    args.bias_correction = bias_correction
    args.ceq = ceq
    args.compress = compress
    args.mode = mode
    args.smooth_quant = smooth_quant
    args.calibration_type = calibration_type
    args.batch_size = batch_size
    args.calibration_ds_extn = calibration_ds_extn
    args.device_type = device_type
    args.input_resource = input_resource
    args.pipeline_name = pipeline_name
    args.config_yaml = config_yaml
    args.no_box_decode = no_box_decode
    args.rtsp_src = rtsp_src
    args.host_ip = host_ip
    args.host_port = host_port
    args.detection_threshold = detection_threshold
    args.nms_iou_threshold = nms_iou_threshold
    args.topk = topk
    args.labels_file = labels_file
    args.num_classes = num_classes
    args.step = step
    if not config_yaml:
        missing = []
        if not model_path:
            missing.append("--model-path")
        if not model_name:
            missing.append("--model-name")
        if not pipeline_name:
            missing.append("--pipeline-name")
        if missing:
            typer.echo(
                f"{' '.join(missing)} parameter{'s are' if len(missing) > 1 else ' is'} missing"
            )
            raise typer.Exit(code=1)
    else:
        if not Path(config_yaml).exists():
            typer.echo(f"{config_yaml} doesn't exist.. Exiting..")
            typer.Exit(code=1)
        else:
            logging.info(
                f"Overriding the params from commandline with parameters from {config_yaml}"
            )
            with open(config_yaml, "r") as config:
                config_data = yaml.safe_load(config)
                build_tables_from_yaml(config_yaml)

            args.pipeline_name = config_data.get("pipeline_name", "MyPipeline")
            # Overriding model params
            args.model_path = config_data.get("model_params", {}).get(
                "model_path", model_path
            )
            args.model_name = config_data.get("model_params", {}).get(
                "model_name", model_name
            )
            args.post_surgery_model_path = config_data.get("model_params", {}).get(
                "post_surgery_model_path", post_surgery_model_path
            )
            args.model_type = config_data.get("model_params", {}).get(
                "model_type", model_type
            )
            args.input_width = config_data.get("model_params", {}).get(
                "input_width", input_width
            )
            args.input_height = config_data.get("model_params", {}).get(
                "input_height", input_height
            )

            # Overriding compilation params
            args.compilation_result_dir = config_data.get("compilation_params", {}).get(
                "compilation_result_dir", compilation_result_dir
            )
            args.compiler = config_data.get("compilation_params", {}).get(
                "compiler", compiler
            )
            args.batch_size = config_data.get("compilation_params", {}).get(
                "batch_size", batch_size
            )
            args.batch_size = config_data.get("compilation_params", {}).get(
                "batch_size", batch_size
            )
            args.device_type = config_data.get("compilation_params", {}).get(
                "device_type", device_type
            )
            # Calibration params
            args.calibration_data_path = (
                config_data.get("compilation_params", {})
                .get("calibration_params", {})
                .get("calibration_data_path", calibration_data_path)
            )
            args.calibration_ds_extn = (
                config_data.get("compilation_params", {})
                .get("calibration_params", {})
                .get("calibration_ds_extn", calibration_ds_extn)
            )
            args.calibration_samples_count = (
                config_data.get("compilation_params", {})
                .get("calibration_params", {})
                .get("calibration_samples_count", calibration_samples_count)
            )
            args.calibration_type = (
                config_data.get("compilation_params", {})
                .get("calibration_params", {})
                .get("calibration_type", calibration_type)
            )
            # Quantization config
            args.arm_only = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("arm_only", arm_only)
            )

            # Activation quant config
            args.act_asym = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("activation_quant_config", {})
                .get("act_asym", act_asym)
            )
            args.act_per_ch = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("activation_quant_config", {})
                .get("act_per_ch", act_per_ch)
            )
            args.act_bf16 = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("activation_quant_config", {})
                .get("act_bf16", act_bf16)
            )
            args.act_nbits = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("activation_quant_config", {})
                .get("act_nbits", act_nbits)
            )

            # Weight quant config
            args.wt_asym = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("weight_quant_config", {})
                .get("wt_asym", wt_asym)
            )
            args.wt_per_ch = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("weight_quant_config", {})
                .get("wt_per_ch", wt_per_ch)
            )
            args.wt_bf16 = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("weight_quant_config", {})
                .get("wt_bf16", wt_bf16)
            )
            args.wt_nbits = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("weight_quant_config", {})
                .get("wt_nbits", wt_nbits)
            )

            args.bias_correction = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("bias_correction", bias_correction)
            )
            args.ceq = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("ceq", ceq)
            )
            args.compress = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("compress", compress)
            )
            args.mode = (
                config_data.get("compilation_params", {})
                .get("quant_config", {})
                .get("mode", mode)
            )

            # Extra params
            args.no_box_decode = config_data.get("extras", {}).get(
                "no_box_decode", no_box_decode
            )
            args.rtsp_src = config_data.get("extras", {}).get("rtsp_src", rtsp_src)
            args.host_ip = config_data.get("extras", {}).get("host_ip", host_ip)
            args.host_port = config_data.get("extras", {}).get("host_port", host_port)
            args.input_resource = config_data.get("extras", {}).get(
                "input_resource", input_resource
            )

            args.detection_threshold = config_data.get("extras", {}).get(
                "detection_threshold", detection_threshold
            )
            args.nms_iou_threshold = config_data.get("extras", {}).get(
                "nms_iou_threshold", nms_iou_threshold
            )
            args.topk = config_data.get("extras", {}).get("topk", topk)
            args.num_classes = config_data.get("extras", {}).get(
                "num_classes", num_classes
            )
            args.labels_file = config_data.get("extras", {}).get(
                "labels_file", labels_file
            )

    # Any specifi step to run
    args.step = step

    # Calling main function now
    main(args=args)


if __name__ == "__main__":
    model_to_pipeline_app()
