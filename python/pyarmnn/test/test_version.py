# Copyright © 2020 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
import os
import importlib


def test_rel_version():
    import pyarmnn._version as v
    importlib.reload(v)
    assert "dev" not in v.__version__
    del v


def test_dev_version():
    import pyarmnn._version as v
    os.environ["PYARMNN_DEV_VER"] = "1"

    importlib.reload(v)

    assert "32.1.0.dev1" == v.__version__

    del os.environ["PYARMNN_DEV_VER"]
    del v


def test_arm_version_not_affected():
    import pyarmnn._version as v
    os.environ["PYARMNN_DEV_VER"] = "1"

    importlib.reload(v)

    assert "32.1.0" == v.__arm_ml_version__

    del os.environ["PYARMNN_DEV_VER"]
    del v
