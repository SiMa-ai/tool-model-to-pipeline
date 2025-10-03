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

import json
import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import yaml

from model_to_pipeline.constants.yolo_labels import YOLO_80_LABELS
from model_to_pipeline.utils.process_util import execute_command
from run_sample_pipeline.configs.cli_config import Config


class CreatePeppi:
    """Creates the PePPi pipeline for quick inference"""

    def __init__(self, config: Config):
        self.config = config
        self.pipeline_dir = Path(config.pipeline_name)
        self._channel_mean = []
        self._channel_stddev = []
        self._decode_type = "yolo"

    def get_pipeline_project_yaml(self) -> dict:
        """Returns the project.yaml template"""
        return yaml.safe_load(
            Path(
                Path(os.path.dirname(__file__)) / "configs" / "project.yaml"
            ).read_text()
        )

    def get_channel_mean_and_stddev(self) -> None:
        """Returns channel mean and stddev read from preproc.json of the model"""
        logging.info("Getting channel mean and stddev from preproc.json")

        with TemporaryDirectory() as tempdir:
            shutil.unpack_archive(filename=self.config.model, extract_dir=tempdir)
            with open(Path(tempdir) / "preproc.json", "r") as preprojson:
                data = json.load(preprojson)
                self._channel_mean = data.get("channel_mean", [0.0, 0.0, 0.0])
                self._channel_stddev = data.get("channel_stddev", [1.0, 1.0, 1.0])

            with open(Path(tempdir) / "boxdecoder.json", "r") as boxdecoder:
                self._decode_type = json.load(boxdecoder).get("decode_type", "yolo")

    def prepare_pipeline_dir(self) -> None:
        """Prepares the pipeline directory for PePPi"""
        logging.info(f"Creating the pipeline directory")
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Copying model to pipeline directory: {self.config.model}")
        shutil.copy(self.config.model, self.pipeline_dir)

        # Creating labels.txt
        logging.info(f"Creating labels file")
        if not self.config.labels_file:
            with open(self.pipeline_dir / "labels.txt", "w+") as labels_file:
                labels_file.write("\n".join(YOLO_80_LABELS))
        else:
            shutil.copy(self.config.labels_file, self.pipeline_dir / "labels.txt")

        with open(self.pipeline_dir / "labels.txt", "r") as labels:
            num_classes = len(labels.readlines())

        self.get_channel_mean_and_stddev()

        yaml_data = self.get_pipeline_project_yaml()
        yaml_data.update(
            {
                "source": {"value": self.config.rtsp_src, "name": "rtspsrc"},
                "udp_host": self.config.host_ip,
                "port": self.config.host_port,
                "pipeline": self.config.pipeline_name,
                "Models": [
                    {
                        "name": self.config.pipeline_name,
                        "targz": os.path.basename(self.config.model),
                        "channel_mean": self._channel_mean,
                        "channel_stddev": self._channel_stddev,
                        "num_classes": num_classes,
                        "decode_type": self._decode_type,
                        "normalize": True,
                        "label_file": "labels.txt",
                        "padding_type": "CENTER",
                        "aspect_ratio": True,
                        "detection_threshold": 0.2,
                        "nms_threshold": 0.4,
                    }
                ],
            }
        )
        with open(self.pipeline_dir / "project.yaml", "w+") as project_yaml:
            yaml.dump(yaml_data, project_yaml, default_flow_style=False)

        # Writing main.py
        logging.info("Creating main.py")
        with open(
            Path(os.path.dirname(__file__)) / "configs" / "main_py_template.txt", "r"
        ) as main_py:
            main_py_content = main_py.readlines()

        with open(self.pipeline_dir / "main.py", "w+") as main_py:
            main_py.writelines(main_py_content)

    def create_mpk(self) -> None:
        """Creates project.mpk"""
        logging.info(f"Creating mpk file")
        old_pwd = os.getcwd()
        try:
            os.chdir(self.pipeline_dir)
            op, ec = execute_command(
                command=f"mpk create -s . -d . --peppi --main-file main.py --yaml-file project.yaml"
            )
            list(map(logging.info, op.splitlines()))
            logging.info(f"Exit code: {ec}")
        except Exception as e:
            logging.error(f"Error creating mpk file: {e}")
        finally:
            os.chdir(old_pwd)

    def deploy_pipeline(self) -> None:
        """Deploys the pipeline using mpk deploy"""
        logging.info("Deploying the pipeline")
        op, ec = execute_command(
            command=f"mpk deploy -f {self.pipeline_dir / 'project.mpk'}"
        )
        list(map(logging.info, op.splitlines()))
        logging.info(f"Exit code: {ec}")
