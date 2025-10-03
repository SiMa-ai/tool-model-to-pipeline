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
from collections import OrderedDict


class StepMeta(type):
    """Meta class for steps.

    Args:
        type (type): Instance of the type class.
    """

    registry = OrderedDict()

    def __new__(cls, name, bases, attrs):
        """Create a new step class.

        Args:
            name (str): Name of the step.
            bases (tuple): Base classes of the step.
            attrs (dict): Attributes of the step.

        Returns:
            type: New step class.
        """
        plugin = super().__new__(cls, name, bases, attrs)
        if not name in ["StepBase"]:
            cls.registry[attrs.get("name", name)] = plugin
            cls.registry = OrderedDict(
                sorted(cls.registry.items(), key=lambda item: item[1].sequence)
            )
        return plugin


class StepBase(metaclass=StepMeta):
    """Base class for steps in the pipeline.

    This class serves as a base for all steps in the pipeline, providing a common interface and
    registration mechanism for step plugins.

    Attributes:
        name (str): Name of the step.
    """

    name = None
    sequence = 0

    def run(self, args: argparse.Namespace):
        """Run the step with the given arguments.

        This method should be overridden by subclasses to implement the specific functionality of the step.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")
