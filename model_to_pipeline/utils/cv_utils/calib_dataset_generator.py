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
import glob
import logging
from typing import Any, Callable, Iterable, Tuple
from ev_transforms.transforms import resize
import cv2
import numpy as np

TARGET_HEIGHT = 640
TARGET_WIDTH = 640
MODEL_LAYOUT = None


def get_calibration_dataset_iterator(
    args: argparse.Namespace,
    input_name: str,
    input_shape: Tuple[int, int, int],
    model_layout: str,
    preproc_func: Callable = None,
) -> Iterable[Any]:
    """Gets an iterator over dataset

    Args:
        args (argparse.Namespace): Command line args
        input_name (str): Input name of the model
        input_shape (Tuple[int, int, int]): Input shape
        model_layout (str): Model's layout
        preproc_func (Callable, optional): Function to preproc the dataset.
            Must include function to read from memory . Defaults to None.

    Returns:
        Iterable[Any]: Iterator
    """

    from afe.core.utils import convert_data_generator_to_iterable
    from sima_utils.data.data_generator import DataGenerator
    from afe.ir.defines import InputName

    global TARGET_HEIGHT, TARGET_WIDTH, MODEL_LAYOUT

    TARGET_HEIGHT, TARGET_WIDTH = input_shape[2], input_shape[3]
    logging.info(f"Using target height: {TARGET_HEIGHT}, target width: {TARGET_WIDTH}")
    MODEL_LAYOUT = model_layout
    rgb_input_name = InputName(input_name)
    img_list = glob.glob(args.calibration_data_path + "/*." + args.calibration_ds_extn)
    logging.info(f"Found [{len(img_list)}] number of images in the calibration samples")
    input_generator = DataGenerator({rgb_input_name: img_list})
    preproc_func = preproc_func if preproc_func else read_and_preprocess
    input_generator.map({rgb_input_name: preproc_func})
    length_limit = (
        args.calibration_samples_count
        if args.calibration_samples_count
        else len(img_list)
    )
    return convert_data_generator_to_iterable(
        input_generator, length_limit=length_limit
    )


def read_and_preprocess(image: str) -> np.ndarray:
    """
    Read an image from the given file path and transform it
    to the array format expected by the model.
    """
    global TARGET_HEIGHT, TARGET_WIDTH, MODEL_LAYOUT
    # Load image in HWC layout
    frame = cv2.imread(image)
    if not frame.any():
        raise RuntimeError("Error: Encounter 0 value image")

    # Add batch dimension to use EV transforms
    frame = np.expand_dims(frame, 0)
    logging.info(f"Using target height: {TARGET_HEIGHT}, target width: {TARGET_WIDTH}")
    # Use SiMa's EV transform for resize
    input_image = resize(
        frame,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
        keep_aspect=True,
        deposit_location="center",
        method="linear",
    )

    # Remove batch dimension
    input_image = input_image.squeeze()
    # Convert to RGB layout if model layout is BGR
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Normalization
    input_image = input_image / 255.0
    input_image = input_image.astype(np.float32)
    return input_image
