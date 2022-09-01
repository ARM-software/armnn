# Copyright © 2020 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
import os
import sys
import shutil

import pytest

sys.path.append(os.path.abspath('..'))
from setup import find_armnn, find_includes, linux_gcc_lib_search, check_armnn_version


@pytest.fixture(autouse=True)
def _setup_armnn(tmpdir):
    includes = str(os.path.join(tmpdir, 'include'))
    libs = str(os.path.join(tmpdir, 'lib'))
    os.environ["TEST_ARMNN_INCLUDE"] = includes
    os.environ["TEST_ARMNN_LIB"] = libs
    os.environ["EMPTY_ARMNN_INCLUDE"] = ''

    os.mkdir(includes)
    os.mkdir(libs)

    with open(os.path.join(libs, "libarmnn.so"), "w"):
        pass

    with open(os.path.join(libs, "libarmnnSomeThing1.so"), "w"):
        pass
    with open(os.path.join(libs, "libarmnnSomeThing1.so.1"), "w"):
        pass
    with open(os.path.join(libs, "libarmnnSomeThing1.so.1.2"), "w"):
        pass

    with open(os.path.join(libs, "libarmnnSomeThing2.so"), "w"):
        pass

    with open(os.path.join(libs, "libSomeThing3.so"), "w"):
        pass

    yield

    del os.environ["TEST_ARMNN_INCLUDE"]
    del os.environ["TEST_ARMNN_LIB"]
    del os.environ["EMPTY_ARMNN_INCLUDE"]
    shutil.rmtree(includes)
    shutil.rmtree(libs)


def test_find_armnn(tmpdir):
    lib_names, lib_paths = find_armnn(lib_name='libarmnn*.so',
                                      armnn_libs_env="TEST_ARMNN_LIB",
                                      default_lib_search=("/lib",))
    armnn_includes = find_includes(armnn_include_env="TEST_ARMNN_INCLUDE")

    assert [':libarmnn.so', ':libarmnnSomeThing1.so', ':libarmnnSomeThing2.so'] == sorted(lib_names)
    assert [os.path.join(tmpdir, 'lib')] == lib_paths
    assert [os.path.join(tmpdir, 'include')] == armnn_includes


def test_find_armnn_default_path(tmpdir):
    lib_names, lib_paths = find_armnn(lib_name='libarmnn*.so',
                                      armnn_libs_env="RUBBISH_LIB",
                                      default_lib_search=(os.environ["TEST_ARMNN_LIB"],))
    armnn_includes = find_includes('TEST_ARMNN_INCLUDE')
    assert [':libarmnn.so', ':libarmnnSomeThing1.so', ':libarmnnSomeThing2.so'] == sorted(lib_names)
    assert [os.path.join(tmpdir, 'lib')] == lib_paths
    assert [os.path.join(tmpdir, 'include')] == armnn_includes


def test_not_find_armnn(tmpdir):
    with pytest.raises(RuntimeError) as err:
        find_armnn(lib_name='libarmnn*.so', armnn_libs_env="RUBBISH_LIB",
                   default_lib_search=("/lib",))

    assert 'ArmNN library libarmnn*.so was not found in (\'/lib\',)' in str(err.value)


@pytest.mark.parametrize("env", ["RUBBISH_INCLUDE", "EMPTY_ARMNN_INCLUDE"])
def test_rubbish_armnn_include(tmpdir, env):
    includes = find_includes(armnn_include_env=env)
    assert includes == ['/usr/local/include', '/usr/include']


def test_gcc_serch_path():
    assert linux_gcc_lib_search()


def test_armnn_version():
    check_armnn_version('32.0.0', '32.0.0')


def test_incorrect_armnn_version():
    with pytest.raises(AssertionError) as err:
        check_armnn_version('32.0.0', '32.1.0')

    assert 'Expected ArmNN version is 32.1.0 but installed ArmNN version is 32.0.0' in str(err.value)


def test_armnn_version_patch_does_not_matter():
    check_armnn_version('32.0.0', '32.0.1')
