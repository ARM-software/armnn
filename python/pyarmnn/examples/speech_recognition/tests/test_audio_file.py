# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os

import numpy as np

from context import audio_capture


def test_audio_file(test_data_folder):
    audio_file = os.path.join(test_data_folder, "myVoiceIsMyPassportVerifyMe04.wav")
    capture = audio_capture.AudioCapture(audio_capture.ModelParams(""))
    buffer = capture.from_audio_file(audio_file)
    audio_data = next(buffer)
    assert audio_data.shape == (47712,)
    assert audio_data.dtype == np.float32
