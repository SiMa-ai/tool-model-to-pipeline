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
import logging
from pathlib import Path
import shutil
from model_to_pipeline.compilers.compiler_base import CompilerBase
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box


class CompileYoloGeneric(CompilerBase):
    """Generic Compiler for YOLO models."""

    name = "yolo"

    def compile(self, args: argparse.Namespace):
        """
        Compile the YOLO model using SiMa's SDK.
        This method loads the ONNX model, extracts input and output shapes,
        and prepares the model for quantization and compilation.
        This method is designed to work with the SiMa SDK and assumes that
        the model is in ONNX format. It also supports calibration datasets
        for quantization.

        Args:
            args (argparse.Namespace): Commandline arguments

        Raises:
            RuntimeError: Raises runtime error
        """
        if args.device_type == "both":
            for board in ("davinci", "modalix"):
                self.__compile_for_board(args=args, board=board)
        else:
            self.__compile_for_board(args=args, board=args.device_type)

    def __compile_for_board(self, args: argparse.Namespace, board: str) -> None:
        from afe.apis.defines import (
            QuantizationParams,
            quantization_scheme,
            default_calibration,
            CalibrationMethod,
        )
        from model_to_pipeline.utils.compiler_utils.get_import_params import (
            model_loader,
        )
        from model_to_pipeline.utils.cv_utils.calib_dataset_generator import (
            get_calibration_dataset_iterator,
        )
        from afe.ir.defines import BiasCorrectionType
        from afe.apis.defines import bfloat16_scheme

        # Supported Bias Correction Types
        biascorr_types = {
            "none": BiasCorrectionType.NONE,
            "iterative": BiasCorrectionType.ITERATIVE,
            "regular": BiasCorrectionType.REGULAR,
        }

        # Supported Calibration Types
        calibration_types = [
            "min_max",
            "moving_average",
            "entropy",
            "percentile",
        ]

        logging.info(f"Loading the model")
        model, input_name, input_shape, _, model_layout = model_loader(
            args=args, board=board
        )

        calibration_data = get_calibration_dataset_iterator(
            args=args,
            input_name=input_name,
            input_shape=input_shape,
            model_layout=model_layout,
        )
        # Setting up quantization config from provided args
        quant_configs: QuantizationParams = QuantizationParams(
            calibration_method=(
                CalibrationMethod.from_str(args.calibration_type)
                if args.calibration_type in calibration_types
                else default_calibration()
            ),
            activation_quantization_scheme=quantization_scheme(
                args.act_asym, args.act_per_ch, bits=args.act_nbits
            ),
            weight_quantization_scheme=quantization_scheme(
                args.wt_asym, args.wt_per_ch, bits=args.wt_nbits
            ),
            requantization_mode=args.mode,
            node_names={""},
            custom_quantization_configs=None,
            biascorr_type=biascorr_types.get(
                args.bias_correction, BiasCorrectionType.NONE
            ),
            channel_equalization=args.ceq,
            smooth_quant=args.smooth_quant,
        )
        # End: quantization config setup

        if args.act_bf16:
            logging.info("Enabling bf16 for activation")
            quant_configs = quant_configs.with_activation_quantization(
                bfloat16_scheme()
            )
        if args.wt_bf16:
            logging.info("Enabling bf16 for weight")
            quant_configs = quant_configs.with_weight_quantization(bfloat16_scheme())

        # Cosmetic. Displays the configuration for Quantization
        console = Console()
        console.print("\n[bold underline][/bold underline]")
        table = Table(
            title="SiMa Model and Quantization Details",
            box=box.SQUARE_DOUBLE_HEAD,
            padding=(0, 1, 0, 1),
            show_lines=True,
            row_styles=["cyan", "magenta"],
        )
        table.add_column("Data", justify="left", no_wrap=True, style="bold")
        table.add_column("Value", justify="left", no_wrap=True, style="bold")
        table.add_row(Text("Device Type", style="bold"), Text(board, style="green"))
        table.add_row(
            Text("Model File", style="bold"),
            Text(
                (
                    args.post_surgery_model_path
                    if args.post_surgery_model_path
                    else args.model_path
                ),
                style="bold",
            ),
        )
        table.add_row(
            Text("Model Input Name", style="bold"), Text(input_name, style="bold")
        )
        table.add_row(
            Text("Model Input Shape", style="bold"),
            Text(f"{input_shape}", style="bold"),
        )
        table.add_row(
            Text("Model Layout", style="bold"), Text(model_layout, style="bold")
        )

        for key, val in quant_configs.__dict__.items():
            table.add_row(
                Text(
                    f"{' '.join(list(map(lambda x: x.capitalize(), key.split('_'))))}",
                    style="bold",
                ),  # Split the word and capitalize
                Text(f"{val}", style="bold"),
            )
        console.print(table)
        console.print("\n[bold underline][/bold underline]")

        # Quantization
        logging.info("Starting quantization")
        model_sdk_net = model.quantize(
            calibration_data=calibration_data,
            quantization_config=quant_configs,
            model_name=args.model_name,
            arm_only=args.arm_only,
            automatic_layout_conversion=False,
            log_level=logging.DEBUG,
        )

        # Compilation
        logging.info(f"Compiling for {board}")
        model_sdk_net.compile(
            output_path=Path(args.compilation_result_dir) / board,
            batch_size=args.batch_size,
            log_level=logging.DEBUG,
            compress=args.compress,
        )

        
        compile_model = args.model_name + "_mpk.tar.gz"
        model_file = Path(args.compilation_result_dir) / board / compile_model
        logging.info(f"Renaming the model file: {model_file} to Pipeline Name: {args.pipeline_name}")
        shutil.move(
            src=model_file,
            dst=str(Path(args.compilation_result_dir) / board / args.pipeline_name)
            + ".tar.gz",
        )
