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

import logging
from model_to_pipeline.compilers.compiler_base import CompilerMeta
from model_to_pipeline.pipeline.pipeline_base import PipelineMeta
from model_to_pipeline.steps.steps_base import StepMeta
from model_to_pipeline.surgeons.surgeon_base import SurgeonMeta
import importlib
import pkgutil
import model_to_pipeline.compilers as compilers
import model_to_pipeline.surgeons as surgeons
import model_to_pipeline.steps as steps
import model_to_pipeline.pipeline as pipeline


def load_plugins():
    """
    Load all plugins from the compilers and surgeons directories.
    This function will dynamically import all modules in the specified directories
    and register their classes in the Meta registry.
    """
    logging.info("Loading plugins...")
    plugins_list = [compilers, surgeons, steps, pipeline]
    for plugin in plugins_list:
        for _, modname, _ in pkgutil.iter_modules(
            plugin.__path__, plugin.__name__ + "."
        ):
            importlib.import_module(modname)
    logging.info("Plugins loaded successfully.")


def get_compiler(name: str) -> type | None:
    """
    Retrieve a compiler plugin by its name.

    :param name: The name of the compiler plugin to retrieve.
    :return: The compiler plugin class if found, None otherwise.
    """
    return CompilerMeta.registry.get(name)


def get_surgeon(name: str) -> type | None:
    """
    Retrieve a surgeon plugin by its name.
    :param name: The name of the surgeon plugin to retrieve.
    :return: The surgeon plugin class if found, None otherwise.
    """
    return SurgeonMeta.registry.get(name)


def get_step(name: str) -> type | None:
    """
    Retrieve a step plugin by its name.
    :param name: The name of the step plugin to retrieve.
    :return: The step plugin class if found, None otherwise.
    """
    return StepMeta.registry.get(name)


def get_pipeline(name: str) -> type | None:
    """
    Retrieve a pipeline plugin by its name.
    :param name: The name of the pipeline plugin to retrieve.
    :return: The pipeline plugin class if found, None otherwise.
    """
    return PipelineMeta.registry.get(name)
