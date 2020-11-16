# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

"""Contains helper functions that can be used across the example apps."""

import os
import errno
from pathlib import Path

import numpy as np


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
