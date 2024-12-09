# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os

import numpy as np

from context import audio_utils

labels = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm',
          13: 'n',
          14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y',
          25: 'z',
          26: "'", 27: ' ', 28: '$'}


def test_labels(test_data_folder):
    assert len(labels) == 29
    assert labels[26] == "\'"
    assert labels[27] == r" "
    assert labels[28] == "$"


def test_decoder(test_data_folder):

    output_tensor = os.path.join(test_data_folder, "inference_output.npy")
    encoded = np.load(output_tensor)
    decoded_text = audio_utils.decode(encoded, labels)
    assert decoded_text == "my voice is my pass"
