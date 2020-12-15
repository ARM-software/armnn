# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
import pytest
import os


@pytest.fixture(scope="module")
def test_data_folder(request):
    """
    This fixture returns path to the folder with the shared test resources
    """
    return str(os.path.join(request.fspath.dirname, "test_data"))


def pytest_addoption(parser):
    """
    Adds the program option 'delegate-dir' to pytest
    """
    parser.addoption("--delegate-dir",
                     action="append",
                     help="Directory of the armnn tflite delegate library",
                     required=True)


def pytest_generate_tests(metafunc):
    """
    Makes the program option 'delegate-dir' available to all tests as a function fixture
    """
    if "delegate_dir" in metafunc.fixturenames:
        metafunc.parametrize("delegate_dir", metafunc.config.getoption("delegate_dir"))