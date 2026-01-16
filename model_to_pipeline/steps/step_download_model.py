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


import argparse
import logging
import os
from pathlib import Path
from model_to_pipeline.steps.steps_base import StepBase


class StepDownloadModel(StepBase):
    """Step for performing downloading on the model.

    This step is responsible for modifying the model as per the requirements.
    It uses a surgeon plugin to perform the DownloadModel operation.
    """

    name = "downloadmodel"
    sequence = 1

    def run(self, args: argparse.Namespace) -> bool:
        """Run the Download Model step.

        Args:
            args: Arguments passed to the step.

        Returns:
            bool: True if the Download Model was successful, False otherwise.
        """
        logging.info(f"Checking if model {args.model_path} is available.")

        import shutil
        import ultralytics

        target_path = Path(args.model_path)
        model_file_name = target_path.name
        model_stem = target_path.stem

        # --------------------------------------------------
        # Case 1: Local .pt model → export to ONNX
        # --------------------------------------------------
        if target_path.suffix == ".pt":
            logging.info(f"Exporting local PyTorch model {target_path} to ONNX")

            model = ultralytics.YOLO(str(target_path))
            model.export(format="onnx", opset=13)

            exported_onnx = Path.cwd() / f"{model_stem}.onnx"
            if not exported_onnx.exists():
                logging.error(f"Expected ONNX not found: {exported_onnx}")
                return False

            target_onnx = target_path.with_suffix(".onnx")
            shutil.move(str(exported_onnx), target_onnx)

            args.model_path = str(target_onnx)
            logging.info(f"Exported ONNX model to {args.model_path}")
            return True

        # --------------------------------------------------
        # Case 2: ONNX does not exist → download & export
        # --------------------------------------------------
        if not target_path.exists():
            logging.info(f"Model {target_path} not found. Downloading via Ultralytics.")

            try:
                model = ultralytics.YOLO(model_stem)
                model.export(
                    format="onnx",
                    opset=13,
                    simplify=False,
                    dynamic=False,
                    imgsz=640,
                )

                exported_onnx = Path.cwd() / f"{model_stem}.onnx"
                if not exported_onnx.exists():
                    raise RuntimeError(
                        f"Expected exported ONNX not found: {exported_onnx}"
                    )

                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(exported_onnx), target_path)

                logging.info(f"Model downloaded and exported to {target_path}")
                return True

            except Exception as e:
                logging.exception(
                    f"Failed to download/export model {args.model_name}: {e}"
                )
                return False

        # --------------------------------------------------
        # Case 3: Model already exists
        # --------------------------------------------------
        logging.info(f"Model {target_path} already exists. Skipping download.")
        return True

