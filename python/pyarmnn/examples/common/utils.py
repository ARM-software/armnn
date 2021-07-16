# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contains helper functions that can be used across the example apps."""

import os
import errno
from pathlib import Path

import numpy as np
import pyarmnn as ann


def dict_labels(labels_file_path: str, include_rgb=False) -> dict:
    """Creates a dictionary of labels from the input labels file.

    Args:
        labels_file: Path to file containing labels to map model outputs.
        include_rgb: Adds randomly generated RGB values to the values of the
            dictionary. Used for plotting bounding boxes of different colours.

    Returns:
        Dictionary with classification indices for keys and labels for values.

    Raises:
        FileNotFoundError:
            Provided `labels_file_path` does not exist.
    """
    labels_file = Path(labels_file_path)
    if not labels_file.is_file():
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), labels_file_path
        )

    labels = {}
    with open(labels_file, "r") as f:
        for idx, line in enumerate(f, 0):
            if include_rgb:
                labels[idx] = line.strip("\n"), tuple(np.random.random(size=3) * 255)
            else:
                labels[idx] = line.strip("\n")
        return labels


def prepare_input_tensors(audio_data, input_binding_info, mfcc_preprocessor):
    """
    Takes a block of audio data, extracts the MFCC features, quantizes the array, and uses ArmNN to create the
    input tensors.

    Args:
        audio_data: The audio data to process
        mfcc_instance: the mfcc class instance
        input_binding_info: the model input binding info
        mfcc_preprocessor: the mfcc preprocessor instance
    Returns:
        input_tensors: the prepared input tensors, ready to be consumed by the ArmNN NetworkExecutor
    """

    data_type = input_binding_info[1].GetDataType()
    input_tensor = mfcc_preprocessor.extract_features(audio_data)
    if data_type != ann.DataType_Float32:
        input_tensor = quantize_input(input_tensor, input_binding_info)
    input_tensors = ann.make_input_tensors([input_binding_info], [input_tensor])
    return input_tensors


def quantize_input(data, input_binding_info):
    """Quantize the float input to (u)int8 ready for inputting to model."""
    if data.ndim != 2:
        raise RuntimeError("Audio data must have 2 dimensions for quantization")

    quant_scale = input_binding_info[1].GetQuantizationScale()
    quant_offset = input_binding_info[1].GetQuantizationOffset()
    data_type = input_binding_info[1].GetDataType()

    if data_type == ann.DataType_QAsymmS8:
        data_type = np.int8
    elif data_type == ann.DataType_QAsymmU8:
        data_type = np.uint8
    else:
        raise ValueError("Could not quantize data to required data type")

    d_min = np.iinfo(data_type).min
    d_max = np.iinfo(data_type).max

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            data[row, col] = (data[row, col] / quant_scale) + quant_offset
            data[row, col] = np.clip(data[row, col], d_min, d_max)
    data = data.astype(data_type)
    return data


def dequantize_output(data, output_binding_info):
    """Dequantize the (u)int8 output to float"""

    if output_binding_info[1].IsQuantized():
        if data.ndim != 2:
            raise RuntimeError("Data must have 2 dimensions for quantization")

        quant_scale = output_binding_info[1].GetQuantizationScale()
        quant_offset = output_binding_info[1].GetQuantizationOffset()

        data = data.astype(float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                data[row, col] = (data[row, col] - quant_offset)*quant_scale
    return data
