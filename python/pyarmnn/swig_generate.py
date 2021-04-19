#!/usr/bin/env python3
# Copyright © 2020 Arm Ltd. All rights reserved.
# Copyright 2020 NXP
# SPDX-License-Identifier: MIT
"""This script executes SWIG commands to generate armnn and armnn version wrappers.
This script cannot be moved to ./script dir because it uses find_armnn function from setup.py script.
Both scripts must be in the same folder.
"""
import os
import re
import subprocess
import argparse

from setup import find_includes

__current_dir = os.path.dirname(os.path.realpath(__file__))
__swig_exec = None
__verbose = False

SWIG_EXEC_ENV = "SWIG_EXECUTABLE"


def get_swig_exec(swig_exec_env: str = SWIG_EXEC_ENV):
    """Returns the swig command. Uses either an env variable or the `swig` command
    and verifies it works.

    Args:
        swig_exec_env(str): Env variable pointing to the swig executable.

    Returns:
        str: Path to swig executable.

    Raises:
        RuntimeError: If unable to execute any version of swig.
    """
    swig_exec = os.getenv(swig_exec_env)
    if swig_exec is None:
        swig_exec = "swig"
    if subprocess.Popen([swig_exec, "-version"], stdout=subprocess.DEVNULL):
        return swig_exec
    else:
        raise RuntimeError("Unable to execute swig.")


def check_swig_version(expected_version: str):
    """Checks version of swig.

    Args:
        expected_version(str): String containing expected version.

    Returns:
        bool: True if version is correct, False otherwise
    """
    cmd = subprocess.Popen([__swig_exec, "-version"], stdout=subprocess.PIPE)
    out, _ = cmd.communicate()

    pattern = re.compile(r"(?<=Version ).+(?=$)", re.MULTILINE)
    match = pattern.search(out.decode('utf-8'))

    if match:
        version_string = match.group(0).strip()
        if __verbose:
            print(f"SWIG version: {version_string}")
        return version_string.startswith(expected_version)
    else:
        return False


def generate_wrap(name: str, extr_includes):
    """Generates the python wrapper using swig.

    Args:
        name(str): Name of the wrapper template.
        extr_includes(str): Include paths.

    Raises:
        RuntimeError: If wrapper fails to be generated.
    """
    in_dir = os.path.join(__current_dir, "src", "pyarmnn", "swig")
    out_dir = os.path.join(__current_dir, "src", "pyarmnn", "_generated")
    if __verbose:
        print(f"Generating wrap for {name} ...")
    code = os.system(f"{__swig_exec} -c++ -python -Wall "
        + "-o {} ".format(os.path.join(out_dir, f"{name}_wrap.cpp"))
        + f"-outdir {out_dir} "
        + f"{extr_includes} "
        + f"-I{in_dir} "
        + os.path.join(in_dir, f"{name}.i"))
    if code != 0:
        raise RuntimeError(f"Failed to generate {name} ext.")


if __name__ == "__main__":
    __swig_exec = get_swig_exec()

    # This check is redundant in case CMake is used, it's here for standalone use
    if not check_swig_version('4.'):
        raise RuntimeError("Wrong swig version was found. Expected SWIG version is 4.x.x")

    armnn_includes = find_includes()

    parser = argparse.ArgumentParser("Script to generate SWIG wrappers.")
    parser.add_argument("-v", "--verbose", help="Verbose output.", action="store_true")
    args = parser.parse_args()

    __verbose = args.verbose

    wrap_names = ['armnn_version',
        'armnn',
        'armnn_onnxparser',
        'armnn_tfliteparser',
        'armnn_deserializer']

    for n in wrap_names:
        generate_wrap(n, f"-I{' -I'.join(armnn_includes)} ")
