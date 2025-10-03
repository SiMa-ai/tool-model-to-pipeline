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

from model_to_pipeline.utils.tee import capture_output


class SurgeonMeta(type):
    """Meta class for surgeons.
    This class is responsible for registering all surgeon plugins.
    It collects all subclasses of Surgeon and stores them in a registry.
    Args:
        type (type): Instance of the type class.
    """

    registry = {}

    def __new__(cls, name, bases, attrs):
        plugin = super().__new__(cls, name, bases, attrs)
        if not name in ["Surgeon"]:
            cls.registry[attrs.get("name", name)] = plugin
        return plugin


class SurgeonBase(metaclass=SurgeonMeta):
    name = None

    def run(self, args: argparse.Namespace) -> str:
        """
        Run the plugin with the given arguments.
        This method should be overridden by subclasses.
        """
        ret_val = False
        stderr = ''
        # Implement the logic for performing surgery on models here
        logging.info(f"Performing surgery on {args.model_name} model...")
        args.post_surgery_model_path = args.post_surgery_model_path if args.post_surgery_model_path else args.pipeline_name + '_mod.onnx'
        try:
            _, stderr, ret_val = capture_output(func=self.do_surgery, args=args)
            logging.info(
                f"Model post-surgery: {args.post_surgery_model_path}"
            )
            # Example logic could include model compilation steps
            logging.info(f"Surgery on {args.model_name} model completed successfully.")
            ret_val = True
        except Exception as e:
            logging.error(f"Error during surgery: {e}")
            logging.error("Surgery failed.")
        finally:
            list(map(logging.info, stderr.splitlines()))
            return ret_val

    def do_surgery(self, args: argparse.Namespace) -> None:
        """
        Perform the surgery on the model.
        This method should be overridden by subclasses to implement specific surgery logic.
        """
        raise NotImplementedError("Subclasses must implement this method.")
