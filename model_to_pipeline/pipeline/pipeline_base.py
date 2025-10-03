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

from collections import OrderedDict


class PipelineMeta(type):

    registry = {}

    def __new__(cls, name, bases, attrs):
        """Create a new pipeline class.

        Args:
            name (str): Name of the pipeline.
            bases (tuple): Base classes of the pipeline.
            attrs (dict): Attributes of the pipeline.

        Returns:
            type: New pipeline class.
        """
        plugin = super().__new__(cls, name, bases, attrs)
        if not name in ["PipelineBase"]:
            cls.registry[attrs.get("name", name)] = plugin
            cls.registry = OrderedDict(
                sorted(cls.registry.items(), key=lambda item: item[1].sequence)
            )
        return plugin


class PipelineBase(metaclass=PipelineMeta):
    """Base class for pipelines.

    This class serves as a base for all pipelines, providing a common interface and
    registration mechanism for pipeline plugins.

    Attributes:
        name (str): Name of the pipeline.
    """

    name = None
    sequence = 0

    def run(self, *args, **kwargs):
        """Run the pipeline with the given arguments.

        This method should be overridden by subclasses to implement the specific functionality of the pipeline.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Raises:
            NotImplementedError: If not overridden by subclass.
        """
        raise NotImplementedError("Subclasses must implement this method.")
