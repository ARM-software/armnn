# Copyright © 2020 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
"""
This script executes SWIG commands to generate armnn and armnn version wrappers.
This script cannot be moved to ./script dir because it uses find_armnn function from setup.py script.
Both scripts must be in the same folder.
"""
import os
import re
import subprocess
from pathlib import Path

from setup import find_includes

__current_dir = Path(__file__).parent.absolute()


def check_swig_versoin(version: str):
    proc = subprocess.Popen(["swig -version"],
                            stdout=subprocess.PIPE, shell=True)
    result = proc.communicate()[0].decode("utf-8")

    pattern = re.compile(r"(?<=Version ).+(?=$)", re.MULTILINE)
    match = pattern.search(result)

    if match:
        version_string = match.group(0).strip()
        print(f"Swig version = {version_string}")
        return version_string.startswith(version)
    else:
        print(f"Failed to find version string in 'swig -version':\n {result}")
        return False


def generate_wrap(name, extr_includes):
    print(f'\nGenerating wrappers for {name}\n')

    code = os.system(f"swig -v -c++ -python"
                     f" -Wall"
                     f" -o {__current_dir}/src/pyarmnn/_generated/{name}_wrap.cpp "
                     f"-outdir {__current_dir}/src/pyarmnn/_generated "
                     f"{extr_includes} "
                     f"-I{__current_dir}/src/pyarmnn/swig "
                     f"{__current_dir}/src/pyarmnn/swig/{name}.i")

    if code != 0:
        raise RuntimeError(f"Failed to generate {name} ext.")


if __name__ == "__main__":
    if not check_swig_versoin('4.'):
        raise RuntimeError("Wrong swig version was found. Expected SWIG version is 4.x.x")

    armnn_includes = find_includes()

    generate_wrap('armnn_version', f"-I{'-I'.join(armnn_includes)} ")
    generate_wrap('armnn', f"-I{'-I'.join(armnn_includes)} ")

    generate_wrap('armnn_caffeparser', f"-I{'-I'.join(armnn_includes)} ")
    generate_wrap('armnn_onnxparser', f"-I{'-I'.join(armnn_includes)} ")
    generate_wrap('armnn_tfparser', f"-I{'-I'.join(armnn_includes)} ")
    generate_wrap('armnn_tfliteparser', f"-I{'-I'.join(armnn_includes)} ")


