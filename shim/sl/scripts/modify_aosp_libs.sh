#!/usr/bin/env bash

#
# Copyright © 2022 ARM Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

AOSP_WORKING_DIR=$1

if [ "$#" -ne 1 ]; then

    echo "Usage: This script must be passed a single parameter which is a path "
    echo "       to an existing directory where the AOSP repo's have been cloned."
    echo "Error: No working directory path parameter provided."
    exit 1
fi
if [ ! -d "$1" ]; then

    echo "Usage: This script must be passed a single parameter which is a path "
    echo "       to an existing directory where the AOSP repo's have been cloned."
    echo "Error: Working directory path provided is not a directory."
    exit 1
fi

SCRIPT_PATH=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "SCRIPT_PATH= ${SCRIPT_PATH}"

pushd "${AOSP_WORKING_DIR}/system/libbase/"
  echo "Applying libbase logging.cpp patch"
  git apply "${SCRIPT_PATH}/libbase_logging_cpp.patch"
popd

pushd "${AOSP_WORKING_DIR}/packages/modules/NeuralNetworks/"
  echo "Applying NeuralNetworks patch"
  git apply "${SCRIPT_PATH}/NeuralNetworks.patch"
popd
