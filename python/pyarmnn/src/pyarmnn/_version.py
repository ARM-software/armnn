# Copyright © 2020 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
import os

version_info = (32, 1, 0)

__dev_version_env = os.getenv("PYARMNN_DEV_VER", "")

if __dev_version_env:
    __dev_version = "dev0"
    try:
        __dev_version = "dev{}".format(int(__dev_version_env))
    except ValueError:
        __dev_version = str(__dev_version_env)

    version_info = (*version_info, __dev_version)

__version__ = '.'.join(str(c) for c in version_info)
__arm_ml_version__ = '{}.{}.{}'.format(version_info[0], version_info[1], version_info[2])


def check_armnn_version(installed_armnn_version: str, expected_armnn_version: str = __arm_ml_version__):
    """Compares expected Arm NN version and Arm NN version used to build the package.

    Args:
        installed_armnn_version (str): Arm NN version used to generate the package (e.g. 32.0.0)
        expected_armnn_version (str): Expected Arm NN version

    Returns:
        None
    """
    expected = expected_armnn_version.split('.', 2)
    installed = installed_armnn_version.split('.', 2)

    # only compare major and minor versions, not patch
    assert (expected[0] == installed[0]) and (expected[1] == installed[1]), \
        "Expected ArmNN version is {} but installed ArmNN version is {}".format(expected_armnn_version, installed_armnn_version)
