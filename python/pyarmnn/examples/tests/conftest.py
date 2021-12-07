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

    sys_arch = os.uname().machine
    if sys_arch == "x86_64":
        libarmnn_url = "https://github.com/ARM-software/armnn/releases/download/v21.11/ArmNN-linux-x86_64.tar.gz"
    else:
        libarmnn_url = "https://github.com/ARM-software/armnn/releases/download/v21.11/ArmNN-linux-aarch64.tar.gz"


    files_to_download = ["https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/messi5.jpg",
                         "https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/basketball1.png",
                         "https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/Megamind.avi",
                         "https://github.com/ARM-software/ML-zoo/raw/master/models/object_detection/ssd_mobilenet_v1/tflite_uint8/ssd_mobilenet_v1.tflite",
                         "https://git.mlplatform.org/ml/ethos-u/ml-embedded-evaluation-kit.git/plain/resources/kws/samples/yes.wav",
                         "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-speech-sdk/master/sampledata/audiofiles/myVoiceIsMyPassportVerifyMe04.wav",
                         "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/prediction/1?lite-format=tflite",
                         "https://tfhub.dev/google/lite-model/magenta/arbitrary-image-stylization-v1-256/int8/transfer/1?lite-format=tflite",
                         libarmnn_url
                         ]

    for file in files_to_download:
        path, filename = ntpath.split(file)
        if filename == '1?lite-format=tflite' and 'prediction' in file:
            filename = 'style_predict.tflite'
        elif filename == '1?lite-format=tflite' and 'transfer' in file:
            filename = 'style_transfer.tflite'
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print("\nDownloading test file: " + file_path + "\n")
            urllib.request.urlretrieve(file, file_path)

    path, filename = ntpath.split(libarmnn_url)
    file_path = os.path.join(data_dir, filename)
    os.system(f"tar -xvzf {file_path} -C {data_dir} ")

    return data_dir
