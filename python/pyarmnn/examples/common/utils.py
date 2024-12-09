# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contains helper functions that can be used across the example apps."""

import os
import errno
from pathlib import Path

import numpy as np
import datetime


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


def prepare_input_data(audio_data, input_data_type, input_quant_scale, input_quant_offset, mfcc_preprocessor):
    """
    Takes a block of audio data, extracts the MFCC features, quantizes the array, and uses ArmNN to create the
    input tensors.

    Args:
        audio_data: The audio data to process
        mfcc_instance: The mfcc class instance
        input_data_type: The model's input data type
        input_quant_scale: The model's quantization scale
        input_quant_offset: The model's quantization offset
        mfcc_preprocessor: The mfcc preprocessor instance
    Returns:
        input_data: The prepared input data
    """

    input_data = mfcc_preprocessor.extract_features(audio_data)
    if input_data_type != np.float32:
        input_data = quantize_input(input_data, input_data_type, input_quant_scale, input_quant_offset)
    return input_data


def quantize_input(data, input_data_type, input_quant_scale, input_quant_offset):
    """Quantize the float input to (u)int8 ready for inputting to model."""
    if data.ndim != 2:
        raise RuntimeError("Audio data must have 2 dimensions for quantization")

    if (input_data_type != np.int8) and (input_data_type != np.uint8):
        raise ValueError("Could not quantize data to required data type")

    d_min = np.iinfo(input_data_type).min
    d_max = np.iinfo(input_data_type).max

    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            data[row, col] = (data[row, col] / input_quant_scale) + input_quant_offset
            data[row, col] = np.clip(data[row, col], d_min, d_max)
    data = data.astype(input_data_type)
    return data


def dequantize_output(data, is_output_quantized, output_quant_scale, output_quant_offset):
    """Dequantize the (u)int8 output to float"""

    if is_output_quantized:
        if data.ndim != 2:
            raise RuntimeError("Data must have 2 dimensions for quantization")

        data = data.astype(float)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                data[row, col] = (data[row, col] - output_quant_offset)*output_quant_scale
    return data


class Profiling:
    def __init__(self, enabled: bool):
        self.m_start = 0
        self.m_end = 0
        self.m_enabled = enabled

    def profiling_start(self):
        if self.m_enabled:
            self.m_start = datetime.datetime.now()

    def profiling_stop_and_print_us(self, msg):
        if self.m_enabled:
            self.m_end = datetime.datetime.now()
            period = self.m_end - self.m_start
            period_us = period.seconds * 1_000_000 + period.microseconds
            print(f'Profiling: {msg} : {period_us:,} microSeconds')
            return period_us
        return 0
