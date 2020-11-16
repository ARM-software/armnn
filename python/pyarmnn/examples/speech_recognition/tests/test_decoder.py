# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os

import numpy as np

from context import common_utils
from context import audio_utils


def test_labels(test_data_folder):
    labels_file = os.path.join(test_data_folder, "wav2letter_labels.txt")
    labels = common_utils.dict_labels(labels_file)
    assert len(labels) == 29
    assert labels[26] == "\'"
    assert labels[27] == r" "
    assert labels[28] == "$"


def test_decoder(test_data_folder):
    labels_file = os.path.join(test_data_folder, "wav2letter_labels.txt")
    labels = common_utils.dict_labels(labels_file)

    output_tensor = os.path.join(test_data_folder, "inf_out.npy")
    encoded = np.load(output_tensor)
    decoded_text = audio_utils.decode(encoded, labels)
    assert decoded_text == "and he walkd immediately out of the apartiment by anothe"
