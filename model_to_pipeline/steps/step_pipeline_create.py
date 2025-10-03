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
from model_to_pipeline.steps.steps_base import StepBase
from model_to_pipeline.utils.loader import get_pipeline


class StepPipelineCreate(StepBase):
    """Step for performing PipelineCreate on the model.

    This step is responsible for modifying the model as per the requirements.
    It uses a surgeon plugin to perform the Pipeline Create operation.
    """

    name = "pipelinecreate"
    sequence = 5

    def run(self, args: argparse.Namespace) -> bool:
        """Run the Pipeline Create step.

        Args:
            args: Arguments passed to the step.

        Returns:
            bool: True if the Pipeline Create was successful, False otherwise.
        """
        pipeline_create = get_pipeline("create_pipeline")
        if pipeline_create:
            instance = pipeline_create()
            result = instance.run(args)
            return result
        else:
            raise ValueError(f"Plugin PipelineCreate for {args.model_name} not found.")
