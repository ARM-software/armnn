#!/bin/bash
#
# Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Script which stores common variables and paths used by setup-armnn.sh and build-armnn.sh

# shellcheck disable=SC2034
# SC2034: false positives for variables appear unused - variables are used in setup-armnn.sh and build-armnn.sh

set -o nounset  # Catch references to undefined variables.
set -o pipefail # Catch non zero exit codes within pipelines.
set -o errexit  # Catch and propagate non zero exit codes.

# ROOT_DIR is the directory in which a script is called from
ROOT_DIR=$(pwd)
SOURCE_DIR="$ROOT_DIR"/source
BUILD_DIR="$ROOT_DIR"/build

# Host architecture e.g. x86_64, aarch64
HOST_ARCH=$(uname -m)

# Number of online cores on host
NUM_THREADS=$(getconf _NPROCESSORS_ONLN)

# Validate common user-defined options
# shellcheck source=validation.sh
source "$rel_path"/validation.sh

# target_arch supplied as command line arg
TARGET_ARCH="$target_arch"

NATIVE_BUILD=0
if [ "$TARGET_ARCH" == "$HOST_ARCH" ]; then
  NATIVE_BUILD=1
elif [ "$TARGET_ARCH" == "aarch64" ]; then
  if [ "$HOST_ARCH" == "arm64" ]; then
    NATIVE_BUILD=1
  fi
fi

AARCH64_COMPILER_FLAGS+="CC=/usr/bin/aarch64-linux-gnu-gcc CXX=/usr/bin/aarch64-linux-gnu-g++ "
if [ "$HOST_ARCH" == "arm64" ]; then
  AARCH64_COMPILER_FLAGS+="CC=/usr/bin/clang CXX=/usr/bin/clang++ "
fi

# NDK
NDK_VERSION=26b
NDK_SRC="$SOURCE_DIR"/android-ndk-r"$NDK_VERSION"

# ANDROID
ANDROID_API_VERSION=30
ANDROID_ARM_ARCH="arm64-v8a"
ANDROID64_x86_TOOLCHAIN="$NDK_SRC/toolchains/llvm/prebuilt/linux-x86_64/"
ANDROID64_COMPILER_FLAGS="CC="$ANDROID64_x86_TOOLCHAIN"/bin/aarch64-linux-android"$ANDROID_API_VERSION"-clang \
                           CXX="$ANDROID64_x86_TOOLCHAIN"/bin/aarch64-linux-android"$ANDROID_API_VERSION"-clang++ "

# Flatbuffers
FLATBUFFERS_VERSION=24.3.25
FLATBUFFERS_SRC="$SOURCE_DIR"/flatbuffers-"$FLATBUFFERS_VERSION"
FLATBUFFERS_BUILD_ROOT="$BUILD_DIR"/flatbuffers
FLATBUFFERS_BUILD_TARGET="$FLATBUFFERS_BUILD_ROOT"/"$TARGET_ARCH"_build
FLATBUFFERS_BUILD_HOST="$FLATBUFFERS_BUILD_ROOT"/"$HOST_ARCH"_build # Location of flatc compiler

# Tensorflow
TENSORFLOW_VERSION=v2.19.0 # v2.19.0
TENSORFLOW_SRC="$SOURCE_DIR"/tensorflow
TFLITE_SRC="$TENSORFLOW_SRC"/tensorflow/lite
SCHEMA_SRC="$TENSORFLOW_SRC"/tensorflow/compiler/mlir/lite/schema/schema.fbs

# TF Lite Schema
FLATC="$FLATBUFFERS_BUILD_HOST"/bin/flatc
TFLITE_BUILD_ROOT="$BUILD_DIR"/tflite # Generated TF Lite Schema location
TFLITE_BUILD_TARGET="$TFLITE_BUILD_ROOT"/"$TARGET_ARCH"_build

# Protobuf
PROTOBUF_VERSION=21.9
PROTOBUF_SRC="$SOURCE_DIR"/protobuf-"$PROTOBUF_VERSION"
PROTOBUF_BUILD_ROOT="$BUILD_DIR"/protobuf
PROTOBUF_BUILD_HOST="$PROTOBUF_BUILD_ROOT"/"$HOST_ARCH"_build
PROTOCOL_COMPILER_HOST="$PROTOBUF_BUILD_HOST"/bin/protoc
PROTOBUF_BUILD_TARGET="$PROTOBUF_BUILD_ROOT"/"$TARGET_ARCH"_build
if [ "$osname" == "Darwin" ]; then
  PROTOBUF_LIBRARY_TARGET="$PROTOBUF_BUILD_HOST"/lib/libprotobuf.dylib
else
  PROTOBUF_LIBRARY_TARGET="$PROTOBUF_BUILD_TARGET"/lib/libprotobuf.so.23.0.0
fi
PROTOBUF_ANDROID_LIB_TARGET="$PROTOBUF_BUILD_TARGET"/lib/libprotobuf.so

# ONNX
ONNX_VERSION=1.6.0
ONNX_SRC="$SOURCE_DIR"/onnx-"$ONNX_VERSION"
ONNX_BUILD_TARGET="$BUILD_DIR"/onnx/"$TARGET_ARCH"_build

# Arm NN / ACL
ARMNN_SRC="$SOURCE_DIR"/armnn
ACL_SRC="$SOURCE_DIR"/acl

# Check if directory at $1 is a repository or not
check_if_repository()
{
  pushd "$1" > /dev/null

  if [ "$(git rev-parse --is-inside-work-tree 2> /dev/null)" ]; then
    popd > /dev/null
    return 0
  else
    popd > /dev/null
    return 1
  fi
}
