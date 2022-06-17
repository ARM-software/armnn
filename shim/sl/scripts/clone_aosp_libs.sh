#!/usr/bin/env bash

#
# Copyright © 2022 ARM Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

AOSP_WORKING_DIR=$1

if [ "$#" -ne 1 ]; then
    echo "Usage: This script must be passed a single parameter which is a path "
    echo "       to an existing directory where the AOSP repo's will be cloned into."
    echo "Error: No working directory path parameter provided."
    exit 1
fi
if [ ! -d "$1" ]; then
    echo "Usage: This script must be passed a single parameter which is a path "
    echo "       to an existing directory where the AOSP repo's will be cloned into."
    echo "Error: Working directory path provided is not a directory."
    exit 1
fi

echo "AOSP_WORKING_DIR = $AOSP_WORKING_DIR"

# NNAPI SUPPORT (SHA's for each repo taken from master branch 25/03/22)
git clone https://android.googlesource.com/platform/packages/modules/NeuralNetworks/ "${AOSP_WORKING_DIR}/packages/modules/NeuralNetworks"
pushd "${AOSP_WORKING_DIR}/packages/modules/NeuralNetworks"
git checkout 9c2360318a35756addcd5d321a85f9270e0a04da
popd

git clone https://android.googlesource.com/platform/system/core "${AOSP_WORKING_DIR}/system/core/"
pushd "${AOSP_WORKING_DIR}/system/core/"
git  checkout c408ee943a1d9c486e4fac10bee7f76a61c75bab
popd

git clone https://android.googlesource.com/platform/system/libbase "${AOSP_WORKING_DIR}/system/libbase"
pushd "${AOSP_WORKING_DIR}/system/libbase"
git checkout 2d235ac982044ea4985c39a834e2d85c6a8bca8f
popd

git clone https://android.googlesource.com/platform/system/libfmq "${AOSP_WORKING_DIR}/system/libfmq"
pushd "${AOSP_WORKING_DIR}/system/libfmq"
git checkout 331b20e54ddde93785d7688ebb0cdc1cbcf9fd9b
popd

git clone https://android.googlesource.com/platform/frameworks/native "${AOSP_WORKING_DIR}/frameworks/native"
pushd "${AOSP_WORKING_DIR}/frameworks/native"
git checkout fea6523ac18c9d4d40db04c996e833f60ff88489
popd

git clone https://android.googlesource.com/platform/system/logging "${AOSP_WORKING_DIR}/system/logging"
pushd "${AOSP_WORKING_DIR}/system/logging"
git checkout e1a669e529cf5a42cd8b331ca89634bb9dce5cae
popd

git clone https://android.googlesource.com/platform/external/boringssl "${AOSP_WORKING_DIR}/external/boringssl"
pushd "${AOSP_WORKING_DIR}/external/boringssl"
git checkout ebeca38b4ecbe81fdf1d127ef7abb4689722308c
popd

git clone https://android.googlesource.com/platform/external/tensorflow "${AOSP_WORKING_DIR}/external/tensorflow"
pushd "${AOSP_WORKING_DIR}/external/tensorflow"
git checkout a6772d90a9b542ceb50f35f67e1cebf322d8b0d0
popd

git clone https://android.googlesource.com/platform/external/eigen "${AOSP_WORKING_DIR}/external/eigen"
pushd "${AOSP_WORKING_DIR}/external/eigen"
git checkout 10f298fc4175c1b8537c674f654a070c871960e5
popd

git clone https://android.googlesource.com/platform/external/ruy "${AOSP_WORKING_DIR}/external/ruy"
pushd "${AOSP_WORKING_DIR}/external/ruy"
git checkout 4377b97cf0850e0a61caa191586ebe68ccbc2abf
popd

git clone https://android.googlesource.com/platform/external/gemmlowp "${AOSP_WORKING_DIR}/external/gemmlowp"
pushd "${AOSP_WORKING_DIR}/external/gemmlowp"
git checkout 689c69e88b91e7bff068e33396f74c0a5b17390e
popd

git clone https://android.googlesource.com/platform/prebuilts/vndk/v29 "${AOSP_WORKING_DIR}/prebuilts/vndk/v29"
pushd "${AOSP_WORKING_DIR}/prebuilts/vndk/v29"
git checkout 5a73511dd91512681df643ce604d36763cd81b0e
popd
