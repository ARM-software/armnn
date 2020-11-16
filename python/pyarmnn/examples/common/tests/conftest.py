# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import ntpath

import urllib.request
import zipfile

import pytest

script_dir = os.path.dirname(__file__)
@pytest.fixture(scope="session")
def test_data_folder(request):
    """
        This fixture returns path to folder with shared test resources among all tests
    """

    data_dir = os.path.join(script_dir, "testdata")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    files_to_download = ["https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/messi5.jpg",
                         "https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/basketball1.png",
                         "https://raw.githubusercontent.com/opencv/opencv/4.0.0/samples/data/Megamind.avi",
                         "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
                         ]

    for file in files_to_download:
        path, filename = ntpath.split(file)
        file_path = os.path.join(data_dir, filename)
        if not os.path.exists(file_path):
            print("\nDownloading test file: " + file_path + "\n")
            urllib.request.urlretrieve(file, file_path)

    # Any unzipping needed, and moving around of files
    with zipfile.ZipFile(os.path.join(data_dir, "coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"), 'r') as zip_ref:
        zip_ref.extractall(data_dir)

    return data_dir
