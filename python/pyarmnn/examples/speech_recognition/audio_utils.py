# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Utilities for speech recognition apps."""

import numpy as np


def decode(model_output: np.ndarray, labels: dict) -> str:
    """Decodes the integer encoded results from inference into a string.

    Args:
        model_output: Results from running inference.
        labels: Dictionary of labels keyed on the classification index.

    Returns:
        Decoded string.
    """
    top1_results = [labels[np.argmax(row)] for row in model_output]
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


def decode_text(is_first_window, labels, output_result):
    """
    Slices the text appropriately depending on the window, and decodes for wav2letter output.
        * First run, take the left context, and inner context.
        * Every other run, take the inner context.
    Stores the current right context, and updates it for each inference. Will get used after last inference.

    Args:
        is_first_window: Boolean to show if it is the first window we are running inference on
        labels: the label set
        output_result: the output from the inference
    Returns:
        current_r_context: the current right context
        text: the current text string, with the latest output decoded and appended
    """
    # For wav2letter with 148 output steps:
    # Left context is index 0-48, inner context 49-99, right context 100-147
    inner_context_start = 49
    inner_context_end = 99
    right_context_start = 100

    if is_first_window:
        # Since it's the first inference, keep the left context, and inner context, and decode
        text = decode(output_result[0][0][0][0:inner_context_end], labels)
    else:
        # Only decode the inner context
        text = decode(output_result[0][0][0][inner_context_start:inner_context_end], labels)

    # Store the right context, we will need it after the last inference
    current_r_context = decode(output_result[0][0][0][right_context_start:], labels)
    return current_r_context, text
