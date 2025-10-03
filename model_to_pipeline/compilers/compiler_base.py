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
import shutil

from model_to_pipeline.utils.tee import capture_output


class CompilerMeta(type):

    registry = {}

    def __new__(cls, name, bases, attrs):
        plugin = super().__new__(cls, name, bases, attrs)
        if name != "CompilerBase":
            cls.registry[attrs.get("name", name)] = plugin
        return plugin


class CompilerBase(metaclass=CompilerMeta):
    name = None

    def run(self, args: argparse.Namespace) -> bool:
        """
        Run the plugin with the given arguments.
        This method should be overridden by subclasses.
        """
        ret_val = False
        stderr = ''
        # Implement the logic for compiling models here
        logging.info(f"Starting compilation for {self.name}")
        try:
            _, stderr, ret_val = capture_output(func=self.compile, args=args)
            # Example logic could include model compilation steps
            logging.info(f"Model {args.model_name} compiled successfully.")
            ret_val = True
        except Exception as e:
            logging.error(f"Error during compilation: {e}")
            logging.error("Compilation failed.")

        finally:
            list(map(logging.info, stderr.splitlines()))
            return ret_val

    def __find_mpk_tar(self) -> Path:
        """Finds mpk tar.gz file recursively in pwd.

        Returns:
            Path: Path to the tar.gz
        """
        from glob import glob

        find = glob("**/*_mpk.tar.gz", recursive=True)
        if len(find) == 0:
            return ""
        return find[0]  # Considering the first find and ingoring others

    def compile(self, args: argparse.Namespace) -> None:
        """
        This method should be overridden by subclasses to implement the actual compilation logic.
        """
        raise NotImplementedError("Subclasses must implement the compile method.")
