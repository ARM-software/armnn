# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Utilities for speech recognition apps."""

import numpy as np


def decode(model_output: np.ndarray, labels: dict) -> list:
    """Decodes the integer encoded results from inference into a string.

    Args:
        model_output: Results from running inference.
        labels: Dictionary of labels keyed on the classification index.

    Returns:
        Decoded string.
    """
    results = [labels[np.argmax(model_output)], model_output[0][0][np.argmax(model_output)]]

    return results


def display_text(text: list):
    """Presents the results on the console.

    Args:
        text: Results of performing ASR on the input audio data.
    """
    print('Classification: %s' % text[0])
    print('Probability: %s' % text[1])
