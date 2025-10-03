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
from model_to_pipeline.constants.calib_dataset import imgIds
import requests


class StepDownloadModel(StepBase):
    """Step for performing downloading on the model.

    This step is responsible for modifying the model as per the requirements.
    It uses a surgeon plugin to perform the DownloadModel operation.
    """

    name = "downloadcalib"
    sequence = 3

    def run(self, args: argparse.Namespace) -> bool:
        """Run the Download Model step.

        Args:
            args: Arguments passed to the step.

        Returns:
            bool: True if the Download Calibration dataset was successful, False otherwise.
        """
        logging.info(f"Checking if Calibration {args.calibration_data_path} is available or not.")

        calib_images_available = os.path.exists(args.calibration_data_path) and len(os.listdir(args.calibration_data_path)) > 0

        if not calib_images_available: #args.calibration_data_path is None or not os.path.exists(args.calibration_data_path):
            logging.info(f"Calibration found {args.calibration_data_path}. Downloading...")
            logging.info(f"Checking for Calibration dataset in local.. ")

            try:
                base_url = "http://images.cocodataset.org/val2017/"
                img_urls = [f"{base_url}{img_id}.jpg" for img_id in imgIds]
                print('img_urls:', img_urls)

                # Create output directory
                output_dir = "/home/docker"
                calib_dir = os.path.join(output_dir, 'calibration_images')
                args.calibration_data_path = calib_dir

                if os.path.exists(calib_dir):
                    if len(os.listdir(calib_dir)) > 0:
                        logging.info(f"Found {len(os.listdir(calib_dir))} images in calibration directory")
                        logging.info(f"✅ Calibration images directory already exists at {calib_dir}")
                        logging.info(f"COCO calibration {args.calibration_data_path} already exists... Skipping downloading")
                        return True
                    else:
                        os.rmdir(calib_dir)
                        logging.warning("Calibration directory exists but is empty. Will re-download images.")

                os.makedirs(calib_dir, exist_ok=True)
                # Download images
                for id, url in enumerate(img_urls):
                    img_data = requests.get(url).content
                    img_filename = os.path.join(calib_dir, f"{imgIds[id]}.jpg")
                    with open(img_filename, 'wb') as f:
                        f.write(img_data)
                logging.info(f"✅ Calibration images downloaded successfully to {calib_dir}")

            except Exception as e:
                logging.error(f"Failed to download calibration dataset: {e}. \n Provide the local path to the calibration dataset using --calibration-data-path")

                return False

        logging.info(f"Calibration {args.calibration_data_path} already exists... Skipping downloading")
        return True
