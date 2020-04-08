# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
import os

version_info = (20, 2, 0)

__dev_version_env = os.getenv("PYARMNN_DEV_VER", "")

if __dev_version_env:
    __dev_version = "dev0"
    try:
        __dev_version = "dev{}".format(int(__dev_version_env))
    except ValueError:
        __dev_version = str(__dev_version_env)

    version_info = (*version_info, __dev_version)

__version__ = '.'.join(str(c) for c in version_info)
__arm_ml_version__ = '2{:03d}{:02d}{:02d}'.format(version_info[0], version_info[1], version_info[2])


def check_armnn_version(installed_armnn_version, expected_armnn_version=__arm_ml_version__):
    expected_armnn_version = expected_armnn_version[:-2]  # cut off minor patch version
    installed_armnn_version = installed_armnn_version[:-2]  # cut off minor patch version
    assert expected_armnn_version == installed_armnn_version, \
        "Expected ArmNN version is {} but installed ArmNN version is {}".format(expected_armnn_version, installed_armnn_version)
