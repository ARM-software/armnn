# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Utilities for speech recognition apps."""

import numpy as np
import pyarmnn as ann


def decode(model_output: np.ndarray, labels: dict) -> str:
    """Decodes the integer encoded results from inference into a string.

    Args:
        model_output: Results from running inference.
        labels: Dictionary of labels keyed on the classification index.

    Returns:
        Decoded string.
    """
    top1_results = [labels[np.argmax(row[0])] for row in model_output]
    return filter_characters(top1_results)


def filter_characters(results: list) -> str:
    """Filters unwanted and duplicate characters.

    Args:
        results: List of top 1 results from inference.

    Returns:
        Final output string to present to user.
    """
    text = ""
    for i in range(len(results)):
        if results[i] == "$":
            continue
        elif i + 1 < len(results) and results[i] == results[i + 1]:
            continue
        else:
            text += results[i]
    return text


def display_text(text: str):
    """Presents the results on the console.

    Args:
        text: Results of performing ASR on the input audio data.
    """
    print(text, sep="", end="", flush=True)


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


def decode_text(is_first_window, labels, output_result):
    """
    Slices the text appropriately depending on the window, and decodes for wav2letter output.
        * First run, take the left context, and inner context.
        * Every other run, take the inner context.
    Stores the current right context, and updates it for each inference. Will get used after last inference

    Args:
        is_first_window: Boolean to show if it is the first window we are running inference on
        labels: the label set
        output_result: the output from the inference
        text: the current text string, to be displayed at the end
    Returns:
        current_r_context: the current right context
        text: the current text string, with the latest output decoded and appended
    """

    if is_first_window:
        # Since it's the first inference, keep the left context, and inner context, and decode
        text = decode(output_result[0][0:472], labels)
    else:
        # Only decode the inner context
        text = decode(output_result[0][49:472], labels)

    # Store the right context, we will need it after the last inference
    current_r_context = decode(output_result[0][473:521], labels)
    return current_r_context, text


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
