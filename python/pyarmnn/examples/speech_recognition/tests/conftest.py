# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT

import os
import ntpath

import urllib.request

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

    files_to_download = ["https://raw.githubusercontent.com/Azure-Samples/cognitive-services-speech-sdk/master"
                         "/sampledata/audiofiles/myVoiceIsMyPassportVerifyMe04.wav"]

    for file in files_to_download:
        path, filename = ntpath.split(file)
        file_path = os.path.join(script_dir, "testdata", filename)
        if not os.path.exists(file_path):
            print("\nDownloading test file: " + file_path + "\n")
            urllib.request.urlretrieve(file, file_path)

    return data_dir
