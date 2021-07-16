# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import ntpath

import urllib.request
import zipfile
import pytest

script_dir = os.path.dirname(__file__)


@pytest.fixture(scope="session")
def test_data_folder():
    """
        This fixture returns path to folder with shared test resources among all tests
    """

    data_dir = os.path.join(script_dir, "testdata")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    files_to_download = ["https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/messi5.jpg",
                         "https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/basketball1.png",
                         "https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/Megamind.avi",
                         "https://github.com/ARM-software/ML-zoo/raw/master/models/object_detection/ssd_mobilenet_v1/tflite_uint8/ssd_mobilenet_v1.tflite",
                         "https://git.mlplatform.org/ml/ethos-u/ml-embedded-evaluation-kit.git/plain/resources/kws/samples/yes.wav",
                         "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-speech-sdk/master/sampledata/audiofiles/myVoiceIsMyPassportVerifyMe04.wav"
                         ]

    for file in files_to_download:
        path, filename = ntpath.split(file)
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print("\nDownloading test file: " + file_path + "\n")
            urllib.request.urlretrieve(file, file_path)


    return data_dir
