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
import os
from typing import Tuple, Any

import model_to_pipeline.utils.onnx_helpers as oh


def model_loader(
    args: argparse.Namespace, board: str
) -> Tuple[Any, str, Tuple[int, int, int], Tuple[int, int, int], str]:
    """Loads the model using onnx.helper.
    Also finds:
    1. Input layer name
    2. Input shape
    3. Layout

    Args:
        args (argparse.Namespace): Command line args
        board (str): Board: davinci or modalix

    Returns:
        Tuple[LoadedNet, str, Tuple[int, int, int, int], Tuple[int, int, int, int], str]: Tuple of Loaded model, input name,
                                                                                input shape, output_shape and layout (NCHW or NHWC)
    """
    from afe.ir.tensor_type import ScalarType
    from afe.apis.loaded_net import LoadedNet
    from afe.load.importers.general_importer import (
        onnx_source,
        # keras_source,
        # tflite_source,
        # pytorch_source,
    )
    from afe.apis.loaded_net import load_model
    from afe.apis.defines import gen1_target, gen2_target

    TARGETS_MAPPER = {"davinci": gen1_target, "modalix": gen2_target}

    assert os.path.exists(
        args.post_surgery_model_path
        if args.post_surgery_model_path
        else args.pipeline_name + "_mod.onnx"
    ), f"Model file {args.post_surgery_model_path if args.post_surgery_model_path else args.pipeline_name + '_mod.onnx'} does not exist."
    # Load the ONNX model
    model = oh.load_model(
        args.post_surgery_model_path
        if args.post_surgery_model_path
        else args.pipeline_name + "_mod.onnx"
    )

    # Extract input and output shapes
    model_input_shapes = {
        inp.name: tuple(dim.dim_value for dim in inp.type.tensor_type.shape.dim)
        for inp in model.graph.input
    }
    output_shapes = {
        out.name: tuple(dim.dim_value for dim in out.type.tensor_type.shape.dim)
        for out in model.graph.output
    }
    output_shape = list(output_shapes.values())[
        0
    ]  # Assuming the first output shape is the one we need
    input_shape = list(model_input_shapes.values())[0]

    # Models importer parameters
    # input shape in format NCHW with N (the batchsize) = 1
    # Considering single input only
    input_name, input_shape, input_type = (
        list(model_input_shapes.keys())[0],
        model_input_shapes[list(model_input_shapes.keys())[0]],
        ScalarType.float32,
    )

    input_shapes_dict = {input_name: input_shape}
    input_types_dict = {input_name: input_type}
    # refer to the SDK User Guide for the specific format
    importer_params = onnx_source(
        (
            args.post_surgery_model_path
            if args.post_surgery_model_path
            else args.pipeline_name + "_mod.onnx"
        ),
        input_shapes_dict,
        input_types_dict,
    )

    loaded_net: LoadedNet = load_model(
        importer_params,
        target=TARGETS_MAPPER.get(board),
    )

    return loaded_net, input_name, input_shape, output_shape, loaded_net._layout
