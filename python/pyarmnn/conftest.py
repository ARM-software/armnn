# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import os
import platform

import pytest

ARCHITECTURES = set("x86_64 aarch64".split())


@pytest.fixture(scope="module")
def data_folder_per_test(request):
    """
        This fixture returns path to folder with test resources (one per test module)
    """

    basedir, script = request.fspath.dirname, request.fspath.basename
    return str(os.path.join(basedir, "testdata", os.path.splitext(script)[0]))


@pytest.fixture(scope="module")
def shared_data_folder(request):
    """
        This fixture returns path to folder with shared test resources among all tests
    """

    return str(os.path.join(request.fspath.dirname, "testdata", "shared"))


@pytest.fixture(scope="function")
def tmpdir(tmpdir):
    """
        This fixture returns path to temp folder. Fixture was added for py35 compatibility
    """

    return str(tmpdir)


def pytest_runtest_setup(item):
    supported_architectures = ARCHITECTURES.intersection(mark.name for mark in item.iter_markers())
    arch = platform.machine()
    if supported_architectures and arch not in supported_architectures:
        pytest.skip("cannot run on platform {}".format(arch))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "aarch64: mark test to run only on aarch64"
    )
    config.addinivalue_line(
        "markers", "x86_64: mark test to run only on x86_64"
    )