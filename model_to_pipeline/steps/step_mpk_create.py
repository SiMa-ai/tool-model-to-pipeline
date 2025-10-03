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
from model_to_pipeline.steps.steps_base import StepBase
from model_to_pipeline.utils.process_util import execute_command
import subprocess, os


class StepMpkCreate(StepBase):
    """Step for creating mpk file from pipeline."""

    name = "mpkcreate"
    sequence = 6

    def run(self, args: argparse.Namespace) -> bool:
        """Run the mpk create step.

        Args:
            args: Arguments passed to the step.

        Returns:
            bool: True if the step was successful, False otherwise.
        """
        if '_simaaisrc' not in args.pipeline_name:
            args.pipeline_name = args.pipeline_name + '_simaaisrc'

        command = f"mpk create -s {os.getcwd()}/{args.pipeline_name} -d {os.getcwd()}/{args.pipeline_name} --clean --board-type {args.device_type if args.device_type != 'both' else 'davinci'}"
        logging.info(f"Executing command to create pipeline: {command}")
        logging.info(f"Python PATH: {os.environ['PATH']}")


        # Optional: if you need to source any environment script
        # env_setup = "source /home/docker/.bashrc"  # or source sdk/env.sh if needed
        env_setup = "source /opt/poky/davinci/5.0.6/environment-setup-cortexa65-poky-linux"

        command = f"unset LD_LIBRARY_PATH && {env_setup} && mpk create -s {os.getcwd()}/{args.pipeline_name} -d {os.getcwd()}/{args.pipeline_name} --clean --board-type {args.device_type if args.device_type != 'both' else 'davinci'}"

        source_path = os.path.abspath(os.path.join(os.getcwd(), args.pipeline_name))
        print("Source path exists:", os.path.exists(source_path))
        print("Source path is dir:", os.path.isdir(source_path))
        print("Source path:", source_path)

        original_env = os.environ.copy()

        # Modify env and run mpk
        env = original_env.copy()
        env.pop("LD_LIBRARY_PATH", None)

        command = f"unset LD_LIBRARY_PATH && {env_setup} && mpk create -s ./ -d ./ --clean --board-type {args.device_type}" #  if args.device_type != 'both' else 'davinci'}"
        logging.info(f"Executing command to create pipeline: {command}")
        op = subprocess.run(command, shell=True, \
                            executable="/bin/bash", \
                            cwd=f"{os.getcwd()}/{args.pipeline_name}/", \
                            capture_output=True, \
                            text=True, env=env)
        
        logging.info(op.stdout)
        logging.error(op.stderr)

        command = f"mpk create -s ./ -d ./ --board-type {args.device_type}" # if args.device_type != 'both' else 'davinci'}"
        logging.info(f"Executing command to create pipeline: {command}")
        op = subprocess.run(command, shell=True, \
                            executable="/bin/bash", \
                            cwd=f"{os.getcwd()}/{args.pipeline_name}/", \
                            capture_output=True, \
                            text=True, env=original_env)
        
        logging.info(op.stdout)
        logging.error(op.stderr)
        ec = op.returncode
        assert ec == 0, f"Failed to create mpk for pipeline {args.pipeline_name}"
        return True
