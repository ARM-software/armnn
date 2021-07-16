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
        This fixture returns path to folder with shared test resources among asr tests
    """

    data_dir = os.path.join(script_dir, "testdata")

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    return data_dir