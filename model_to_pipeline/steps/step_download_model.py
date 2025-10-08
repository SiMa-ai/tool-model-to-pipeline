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
        logging.info(f"Checking if model {args.model_path} is available or not.")
        if not Path(args.model_path).exists():
            logging.info(f"Model {args.model_path} not found. Downloading...")
            import ultralytics # type: ignore

            try:
                model = ultralytics.YOLO(
                    os.path.splitext(os.path.basename(args.model_path))[0]
                )
                model.export(format="onnx", opset=13, simplify=False, dynamic=False, imgsz=640)
                logging.info(
                    f"Model {args.model_name} downloaded successfully to {args.model_path}."
                )
                return True
            except Exception as e:
                logging.error(f"Failed to download model {args.model_name}: {e}")
                return False
        logging.info(f"Model {args.model_path} already exists... Skipping downloading")
        return True
