#!/bin/bash

#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Script which builds Arm NN and ACL
# setup-armnn.sh must be executed in the same directory, before running this script

set -o nounset  # Catch references to undefined variables.
set -o pipefail # Catch non zero exit codes within pipelines.
set -o errexit  # Catch and propagate non zero exit codes.

rel_path=$(dirname "$0") # relative path from where script is executed to script location

build_acl()
{
  cd "$ACL_SRC"

  # $acl_scons_params are additional options provided by the user and will overwrite any previously defined args
  local acl_params="neon=$flag_neon_backend opencl=$flag_cl_backend Werror=0 embed_kernels=1 examples=0 validation_tests=0 benchmark_tests=0 benchmark_examples=0 $acl_scons_params"

  if [ "$flag_debug" -eq 1 ]; then
    acl_params="$acl_params debug=1 asserts=1"
  fi

  local native_flag=""
  if [ "$NATIVE_BUILD" ]; then
    native_flag="build=native"
  fi

  # Force -fPIC so that ACL is suitable for inclusion in Arm NN library
  local extra_cxx_flags="extra_cxx_flags='-fPIC'"

  local compile_flags=""
  local acl_arch=""

  case "$TARGET_ARCH" in
    "aarch64")
      compile_flags+="$AARCH64_COMPILER_FLAGS"
      acl_arch="arch=arm64-v8a"
      ;;

    "x86_64")
      acl_arch="arch=x86_64"
      ;;
  esac

  echo -e "\n***** Building ACL for $TARGET_ARCH *****"

  if [ "$flag_clean" -eq 1 ]; then
    echo -e "\n***** Clean flag detected: removing existing ACL build *****"
    rm -rf "$ACL_BUILD_TARGET"
  fi

  mkdir -p "$ACL_BUILD_TARGET"

  eval "$compile_flags" \
  scons "$native_flag" \
        "$acl_arch" \
        "$acl_params" \
        build_dir="$ACL_BUILD_TARGET" \
        "$extra_cxx_flags" \
        -j "$NUM_THREADS"

  echo -e "\n***** Built ACL for $TARGET_ARCH *****"

  return 0
}

build_armnn()
{
  mkdir -p "$ARMNN_BUILD_TARGET"
  cd "$ARMNN_BUILD_TARGET"

  local build_type="Release"
  if [ "$flag_debug" -eq 1 ]; then
    build_type="Debug"
  fi

  local compile_flags=""

  case "$TARGET_ARCH" in
    "aarch64")
      compile_flags+="$AARCH64_COMPILER_FLAGS"
      ;;
  esac

  if [ "$flag_clean" -eq 1 ]; then
    echo -e "\n***** Clean flag detected: removing existing Arm NN build *****"
    rm -rf "$ARMNN_BUILD_TARGET"
  fi

  echo -e "\n***** Building Arm NN for $TARGET_ARCH *****"

  eval "$compile_flags" \
  cmake -DCMAKE_BUILD_TYPE="$build_type" \
        -DBUILD_CLASSIC_DELEGATE="$flag_tflite_delegate" \
        -DBUILD_TF_LITE_PARSER="$flag_tflite_parser" \
        -DBUILD_ONNX_PARSER="$flag_onnx_parser" \
        -DARMCOMPUTENEON="$flag_neon_backend" \
        -DARMCOMPUTECL="$flag_cl_backend" \
        -DARMNNREF="$flag_ref_backend" \
        -DARMCOMPUTE_ROOT="$ACL_SRC" \
        -DARMCOMPUTE_BUILD_DIR="$ACL_BUILD_TARGET" \
        -DTENSORFLOW_ROOT="$TENSORFLOW_SRC" \
        -DTF_LITE_SCHEMA_INCLUDE_PATH="$TFLITE_BUILD_ROOT" \
        -DTFLITE_LIB_ROOT="$TFLITE_BUILD_TARGET" \
        -DFLATBUFFERS_ROOT="$FLATBUFFERS_BUILD_TARGET" \
        -DFLATC_DIR="$FLATBUFFERS_BUILD_HOST" \
        -DONNX_GENERATED_SOURCES="$ONNX_BUILD_TARGET" \
        -DPROTOBUF_ROOT="$PROTOBUF_BUILD_HOST" \
        -DPROTOBUF_LIBRARY_DEBUG="$PROTOBUF_LIBRARY_TARGET" \
        -DPROTOBUF_LIBRARY_RELEASE="$PROTOBUF_LIBRARY_TARGET" \
        "$armnn_cmake_args" \
        "$ARMNN_SRC"

  make -j "$NUM_THREADS"

  # Copy protobuf library into Arm NN build directory, if ONNX Parser is enabled
  if [ "$flag_onnx_parser" -eq 1 ]; then
    cd "$ARMNN_BUILD_TARGET"
    rm -f libprotobuf.so libprotobuf.so.23 libprotobuf.so.23.0.0
    cp "$PROTOBUF_LIBRARY_TARGET" .
    ln -s libprotobuf.so.23.0.0 ./libprotobuf.so.23
    ln -s libprotobuf.so.23.0.0 ./libprotobuf.so
  fi

  # Copy Arm NN include directory into build output
  cd "$ARMNN_BUILD_TARGET"
  rm -rf include
  cp -r "$SOURCE_DIR"/armnn/include .

  echo -e "\n***** Built Arm NN for $TARGET_ARCH *****"

  local tarball_path="$ROOT_DIR/armnn_$ARMNN_BUILD_DIR_NAME.tar.gz"
  echo -e "\n***** Creating tarball of Arm NN build at $tarball_path *****"

  cd "$ARMNN_BUILD_ROOT"
  rm -f "$tarball_path"
  tar -czf "$tarball_path" "$ARMNN_BUILD_DIR_NAME"

  echo -e "\n***** Created tarball of Arm NN build at $ROOT_DIR/armnn_$ARMNN_BUILD_DIR_NAME.tar.gz *****"
  echo -e "\n***** To extract tarball, run: tar -xzf armnn_$ARMNN_BUILD_DIR_NAME.tar.gz *****\n"

  return 0
}

download_armnn()
{
  cd "$SOURCE_DIR"

  echo -e "\n***** Downloading Arm NN *****"

  rm -rf "$ARMNN_SRC"

  # Latest release branch of Arm NN is checked out by default
  git clone https://github.com/ARM-software/armnn.git armnn

  cd "$ARMNN_SRC"
  local armnn_branch="$(git rev-parse --abbrev-ref HEAD)"

  echo -e "\n***** Arm NN Downloaded: $armnn_branch *****"
}

download_acl()
{
  # First get Arm NN branch so that we can download corresponding ACL tag
  cd "$ARMNN_SRC"
  local armnn_branch="$(git rev-parse --abbrev-ref HEAD)"

  echo -e "\n***** Downloading corresponding ACL version using Arm NN branch: $armnn_branch *****"

  cd "$SOURCE_DIR"

  rm -rf "$ACL_SRC"

  git clone https://github.com/ARM-software/ComputeLibrary.git acl

  # Get corresponding release tag for ACL by parsing release branch number for Arm NN
  local acl_tag=""
  acl_tag="$(echo "$armnn_branch" | tr '\n' ' ' | sed -e 's/[^0-9]/ /g' -e 's/^ *//g' -e 's/ *$//g' | tr -s ' ' | sed 's/ /./g')"

  cd "$ACL_SRC"
  git checkout v"$acl_tag"

  echo -e "\n***** ACL Downloaded: $acl_tag *****"
}

usage()
{
  cat <<EOF
build-armnn.sh - Build Arm NN and ACL
build-armnn.sh [OPTION]...
  --tflite-delegate
    build the Arm NN TF Lite Delegate component
  --tflite-parser
    build the Arm NN TF Lite Parser component
  --onnx-parser
    build the Arm NN ONNX parser component
  --all
    build all Arm NN components listed above
  --target-arch=[aarch64|x86_64]
    specify a target architecture (mandatory)
  --neon-backend
    build Arm NN with the NEON backend (CPU acceleration from ACL)
  --cl-backend
    build Arm NN with the OpenCL backend (GPU acceleration from ACL)
  --ref-backend
    build Arm NN with the reference backend (Should be used for verification purposes only. Does not provide any performance acceleration.)
  --clean
    remove previous Arm NN and ACL build prior to script execution (optional: defaults to off)
  --debug
    build Arm NN (and ACL) with debug turned on (optional: defaults to off)
  --armnn-cmake-args=<ARG LIST STRING>
    provide additional comma-separated CMake arguments string for building Arm NN (optional)
  --acl-scons-params=<PARAM LIST STRING>
    provide additional comma-separated scons parameters string for building ACL (optional)
  --num-threads=<INTEGER>
    specify number of threads/cores to build dependencies with (optional: defaults to number of online CPU cores on host)
  -h, --help
    print brief usage information and exit
  -x
    enable shell tracing in this script

At least one component (i.e. --tflite-delegate, --tflite-parser, --onnx-parser) must be provided or else provide --all to build all Arm NN components.
At least one backend (i.e. --neon-backend, --cl-backend, --ref-backend) must be chosen.
This script must be executed from the same root directory in which setup-armnn.sh was executed from.

The first execution of this script will download the latest release branches of Arm NN and ACL, by default.
Alternatively, place custom/modified repositories named "armnn" and (optionally) "acl" in <ROOT_DIR>/source.
Providing custom "acl" repo is optional since it is only required if backend flags --neon-backend or --cl-backend are chosen.

By default, a tarball tar.gz archive of the Arm NN build will be created in the directory from which this script is called from.

Examples:
Build for aarch64 with all Arm NN components, NEON enabled and OpenCL enabled:
  <PATH_TO>/build-armnn.sh --target-arch=aarch64 --all --neon-backend --cl-backend
Build for aarch64 with TF Lite Delegate, OpenCL enabled and additional ACL scons params:
  <PATH_TO>/build-armnn.sh --target-arch=aarch64 --tflite-delegate --cl-backend --acl-scons-params='compress_kernels=1,benchmark_examples=1'
Setup for aarch64 with all Arm NN dependencies, OpenCL enabled and additional Arm NN cmake args:
  <PATH_TO>/build-armnn.sh --target-arch=aarch64 --all --cl-backend --armnn-cmake-args='-DBUILD_SAMPLE_APP=1,-DBUILD_UNIT_TESTS=0'
EOF
}

# This will catch in validation.sh if not set
target_arch=""

# Default flag values
flag_tflite_delegate=0
flag_tflite_parser=0
flag_onnx_parser=0
flag_neon_backend=0
flag_cl_backend=0
flag_ref_backend=0
flag_clean=0
flag_debug=0

# Empty strings for optional additional args by default
armnn_cmake_args=""
acl_scons_params=""

# If --num-threads is not set, the default NUM_THREADS value in common.sh will be used
num_threads=0

name=$(basename "$0")

# If no options provided, show help
if [ $# -eq 0 ]; then
  usage
  exit 1
fi

args=$(getopt -ohx -l tflite-delegate,tflite-parser,onnx-parser,all,target-arch:,neon-backend,cl-backend,ref-backend,clean,debug,armnn-cmake-args:,acl-scons-params:,num-threads:,help -n "$name"   -- "$@")
eval set -- "$args"
while [ $# -gt 0 ]; do
  if [ -n "${opt_prev:-}" ]; then
    eval "$opt_prev=\$1"
    opt_prev=
    shift 1
    continue
  elif [ -n "${opt_append:-}" ]; then
    if [ -n "$1" ]; then
      eval "$opt_append=\"\${$opt_append:-} \$1\""
    fi
    opt_append=
    shift 1
    continue
  fi
  case $1 in
  --tflite-parser)
    flag_tflite_parser=1
    ;;

  --tflite-delegate)
    flag_tflite_delegate=1
    ;;

  --onnx-parser)
    flag_onnx_parser=1
    ;;

  --all)
    flag_tflite_delegate=1
    flag_tflite_parser=1
    flag_onnx_parser=1
    ;;

  --target-arch)
    opt_prev=target_arch
    ;;

  --neon-backend)
    flag_neon_backend=1
    ;;

  --cl-backend)
    flag_cl_backend=1
    ;;

  --ref-backend)
    flag_ref_backend=1
    ;;

  --clean)
    flag_clean=1
    ;;

  --debug)
    flag_debug=1
    ;;

  --armnn-cmake-args)
    opt_prev=armnn_cmake_args
    ;;

  --acl-scons-params)
    opt_prev=acl_scons_params
    ;;

  --num-threads)
    opt_prev=num_threads
    ;;

  -h | --help)
    usage
    exit 0
    ;;

  -x)
    set -x
    ;;

  --)
    shift
    break 2
    ;;

  esac
  shift 1
done

# shellcheck source=common.sh
source "$rel_path"/common.sh

# Validation of chosen Arm NN backends
if [ "$flag_neon_backend" -eq 0 ] && [ "$flag_cl_backend" -eq 0 ] && [ "$flag_ref_backend" -eq 0 ]; then
  echo -e "\n$name: at least one of flags --neon-backend, --cl-backend or --ref-backend must be set."
  exit 1
fi

if [ "$target_arch" == "x86_64" ]; then
  if [ "$flag_neon_backend" -eq 1 ] || [ "$flag_cl_backend" -eq 1 ]; then
    echo "$name: Accelerated backends --neon-backend and --cl-backend are supported on Arm targets only (x86_64 chosen)."
    exit 1
  fi
fi

# Verify that root source and build directories are present (post execution of setup-armnn.sh)
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "\nERROR: Root source directory does not exist at $SOURCE_DIR"
    echo "Please check that:"
    echo "1. setup-armnn.sh was executed successfully prior to running this script"
    echo "2. This script is being executed in the same directory as setup-armnn.sh"

    exit 1
fi

if [ ! -d "$BUILD_DIR" ]; then
    echo -e "\nERROR: Root build directory does not exist at $BUILD_DIR"
    echo "Please check that:"
    echo "1. setup-armnn.sh was executed successfully prior to running this script"
    echo "2. This script is being executed in the same directory as setup-armnn.sh"

    exit 1
fi

# Download Arm NN if not done already in a previous execution of this script
# Check if Arm NN source directory exists AND that it is a repository (not empty)
if [ -d "$ARMNN_SRC" ] && check_if_repository "$ARMNN_SRC"; then
  echo -e "\n***** Arm NN source repository already located at $ARMNN_SRC. Skipping cloning of Arm NN. *****"
else
  # Download latest release branch of Arm NN
  download_armnn
fi

# Download ACL if not done already in a previous execution of this script
# Only download ACL if backend options --neon-backend and --cl-backend are chosen
if [ "$flag_neon_backend" -eq 1 ] || [ "$flag_cl_backend" -eq 1 ]; then
  # Check if Arm NN source directory exists AND that it is a repository (not empty)
  if [ -d "$ACL_SRC" ] && check_if_repository "$ACL_SRC"; then
    echo -e "\n***** ACL source repository already located at $ACL_SRC. Skipping cloning of ACL. *****"
  else
    # Download latest release branch of ACL
    download_acl
  fi
else
  echo -e "\n***** Backend options --neon-backend and --cl-backend not selected - skipping cloning of ACL *****"
fi

# Adjust output build directory names for Arm NN and ACL if debug is enabled
DEBUG_POSTFIX=""
if [ "$flag_debug" -eq 1 ]; then
  DEBUG_POSTFIX="_debug"
fi

# Replace commas with spaces in additional Arm NN / ACL build args
# shellcheck disable=SC2001
armnn_cmake_args="$(echo "$armnn_cmake_args" | sed 's/,/ /g')"

# shellcheck disable=SC2001
acl_scons_params="$(echo "$acl_scons_params" | sed 's/,/ /g')"

# Directories for Arm NN and ACL build outputs
ARMNN_BUILD_ROOT="$BUILD_DIR"/armnn
ARMNN_BUILD_DIR_NAME="$TARGET_ARCH"_build"$DEBUG_POSTFIX"
ARMNN_BUILD_TARGET="$ARMNN_BUILD_ROOT"/"$ARMNN_BUILD_DIR_NAME"
ACL_BUILD_TARGET="$BUILD_DIR"/acl/"$TARGET_ARCH"_build"$DEBUG_POSTFIX"

echo -e "\nINFO: Displaying configuration information before execution of $name"
echo "     target-arch: $TARGET_ARCH"
echo "       host-arch: $HOST_ARCH"
echo " tflite-delegate: $flag_tflite_delegate"
echo "   tflite-parser: $flag_tflite_parser"
echo "     onnx-parser: $flag_onnx_parser"
echo "    neon-backend: $flag_neon_backend"
echo "      cl-backend: $flag_cl_backend"
echo "     ref-backend: $flag_ref_backend"
echo "           clean: $flag_clean"
echo "           debug: $flag_debug"
echo "armnn-cmake-args: $armnn_cmake_args"
echo "acl-scons-params: $acl_scons_params"
echo "     num-threads: $NUM_THREADS"
echo "  root directory: $ROOT_DIR"
echo "source directory: $SOURCE_DIR"
echo " build directory: $BUILD_DIR"
echo " armnn build dir: $ARMNN_BUILD_TARGET"
echo -e "\nScript execution will begin in 10 seconds..."

sleep 10

if [ "$flag_neon_backend" -eq 1 ] || [ "$flag_cl_backend" -eq 1 ]; then
  build_acl
else
  echo -e "\n***** Skipping ACL build: --neon-backend and --cl-backend not set in options. *****"
fi

build_armnn

exit 0
