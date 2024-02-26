#!/bin/bash
#
# Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Script which downloads and builds Arm NN dependencies
# Perquisite to running build-armnn.sh

set -o nounset  # Catch references to undefined variables.
set -o pipefail # Catch non zero exit codes within pipelines.
set -o errexit  # Catch and propagate non zero exit codes.

rel_path=$(dirname "$0") # relative path from where script is executed to script location

# Figure out platform specific settings
osname=$(uname)
osgetopt=getopt
os_darwin=0
if [ "$osname" == "Darwin" ]; then
  os_darwin=1
  osgetoptsys="/opt/homebrew/opt/gnu-getopt/bin/getopt"
  osgetopthome="$HOME/homebrew/opt/gnu-getopt/bin/getopt"
  if [ -f "$osgetoptsys" ]; then
    echo "gnu-getopt found at: $osgetoptsys"
    osgetopt=$osgetoptsys
  elif [ -f "$osgetopthome" ]; then
    echo "gnu-getopt found at: $osgetopthome"
    osgetopt=$osgetopthome
  else
    echo "Run $rel_path/install-packages.sh and follow the instructions to configure the environment for $osname"
    exit 1
  fi
fi

# Download an archive using wget and extract using tar
# Takes three arguments:
# 1. Name of dependency being downloaded e.g. Flatbuffers
# 2. Link to archive
# 3. Filename given to archive upon downloading
download_and_extract()
{
  cd "$SOURCE_DIR"

  echo -e "\n***** Downloading $1 *****\n"
  wget -O "$3" "$2"

  echo -e "\n***** Extracting archive *****"
  tar -xzf "$3"

  echo -e "\n***** Removing archive *****"
  rm "$3"

  echo -e "\n***** $1 downloaded *****"
}

download_androidndk()
{
  cd "$SOURCE_DIR"
  echo -e "\n***** Downloading Android NDK *****\n"
  wget https://dl.google.com/android/repository/android-ndk-r26b-linux.zip
  echo -e "\n***** Extracting archive *****"
  unzip android-ndk-r26b-linux.zip
  echo -e "\n***** Removing archive *****"
  rm android-ndk-r26b-linux.zip
  echo -e "\n***** Android NDK downloaded *****"
}

download_protobuf()
{
  download_and_extract \
    "Protobuf" \
    "https://github.com/protocolbuffers/protobuf/releases/download/v$PROTOBUF_VERSION/protobuf-all-$PROTOBUF_VERSION.tar.gz" \
    "protobuf-all-$PROTOBUF_VERSION.tar.gz"
}

build_protobuf()
{
  local native_build=$1
  local build_dir="$PROTOBUF_BUILD_TARGET"
  local cmake_flags=""
  local target_arch="$TARGET_ARCH"
  local additional_cmds=""

  if [ "$native_build" -eq 0 ]; then
    mkdir -p "$PROTOBUF_BUILD_TARGET"
    additional_cmds+="--with-protoc=$PROTOCOL_COMPILER_HOST "
    if [ "$TARGET_ARCH" == "aarch64" ]; then
      cmake_flags+="$AARCH64_COMPILER_FLAGS"
      additional_cmds+="--host=aarch64-linux "
    fi
    if [ "$TARGET_ARCH" == "android64" ]; then
      additional_cmds+="--host=aarch64-linux-android "
      cmake_flags+="$ANDROID64_COMPILER_FLAGS"
    fi
  else
    target_arch="$HOST_ARCH"
    mkdir -p "$PROTOBUF_BUILD_HOST"
    build_dir="$PROTOBUF_BUILD_HOST"
  fi

  echo -e "\n***** Building Protobuf for $target_arch ***** "

  cd "$PROTOBUF_BUILD_ROOT"

  # Cleanup any previous cmake files, except actual builds which we keep
  find . -mindepth 1 -name "*_build" -prune -o -exec rm -rf {} +

  if [ "$native_build" -eq 0 ] && [ "$TARGET_ARCH" == "android64" ]; then
      eval "$cmake_flags"
      cmake -DCMAKE_ANDROID_NDK="$NDK_SRC"  \
            -DCMAKE_SYSTEM_NAME=Android \
            -DCMAKE_SYSTEM_VERSION="$ANDROID_API_VERSION" \
            -DCMAKE_ANDROID_ARCH_ABI="$ANDROID_ARM_ARCH" \
            -DCMAKE_CXX_FLAGS=--std=c++14 \
            -Dprotobuf_BUILD_TESTS=OFF \
            -Dprotobuf_BUILD_SHARED_LIBS=ON \
            -Dprotobuf_WITH_ZLIB=OFF \
            -DCMAKE_BUILD_TYPE=Release \
            $PROTOBUF_SRC/cmake/
      make libprotobuf -j "$NUM_THREADS"
      cmake -DCMAKE_INSTALL_PREFIX=$PROTOBUF_BUILD_TARGET -DCOMPONENT=libprotobuf -P cmake_install.cmake
      cmake -DCMAKE_INSTALL_PREFIX=$PROTOBUF_BUILD_TARGET -DCOMPONENT=protobuf-headers -P cmake_install.cmake
  else
      eval "$cmake_flags" \
      "$PROTOBUF_SRC"/configure --prefix="$build_dir" "$additional_cmds"
      make install -j "$NUM_THREADS"
  fi

  echo -e "\n***** Protobuf built for $target_arch ***** "
}

download_flatbuffers()
{
  download_and_extract \
    "Flatbuffers" \
    "https://github.com/google/flatbuffers/archive/v$FLATBUFFERS_VERSION.tar.gz" \
    "flatbuffers-$FLATBUFFERS_VERSION.tar.gz"
}

build_flatbuffers()
{
  local native_build=$1
  local build_dir="$FLATBUFFERS_BUILD_TARGET"
  local target_arch="$TARGET_ARCH"

  local cmake_flags="CXXFLAGS=-fPIC "

  if [ "$native_build" -eq 0 ]; then
    mkdir -p "$FLATBUFFERS_BUILD_TARGET"
    if [ "$TARGET_ARCH" == "aarch64" ]; then
      cmake_flags+="$AARCH64_COMPILER_FLAGS"
    fi
    if [ "$TARGET_ARCH" == "android64" ]; then
      cmake_flags+="$ANDROID64_COMPILER_FLAGS"
    fi
  else
    target_arch="$HOST_ARCH"
    mkdir -p "$FLATBUFFERS_BUILD_HOST"
    build_dir="$FLATBUFFERS_BUILD_HOST"
    if [ "$os_darwin" -eq 1 ]; then
      cmake_flags+="$AARCH64_COMPILER_FLAGS"
    fi
  fi

  echo -e "\n***** Building flatbuffers for $target_arch *****"

  mkdir -p "$FLATBUFFERS_BUILD_ROOT"
  cd "$FLATBUFFERS_BUILD_ROOT"

  # Cleanup any previous cmake files, except actual builds which we keep
  find . -mindepth 1 -name "*_build" -prune -o -exec rm -rf {} +

  if [ "$native_build" -eq 0 ] && [ "$TARGET_ARCH" == "android64" ]; then
    eval "$cmake_flags" \
    cmake -DCMAKE_ANDROID_NDK="$NDK_SRC" \
          -DCMAKE_SYSTEM_NAME=Android \
          -DCMAKE_SYSTEM_VERSION="$ANDROID_API_VERSION" \
          -DCMAKE_ANDROID_ARCH_ABI="$ANDROID_ARM_ARCH" \
          -DCMAKE_CXX_FLAGS=--std=c++14 \
          -DFLATBUFFERS_BUILD_FLATC=0 \
          -DCMAKE_INSTALL_PREFIX:PATH="$build_dir" \
          -DCMAKE_BUILD_TYPE=Release \
          -DFLATBUFFERS_BUILD_TESTS=0 \
          "$FLATBUFFERS_SRC"
  else
    eval "$cmake_flags" \
    cmake -DFLATBUFFERS_BUILD_FLATC="$native_build" \
          -DCMAKE_INSTALL_PREFIX:PATH="$build_dir" \
          -DFLATBUFFERS_BUILD_TESTS=0 \
          "$FLATBUFFERS_SRC"
  fi
  make all install -j "$NUM_THREADS"

  echo -e "\n***** Built flatbuffers for $target_arch *****"
}

download_tensorflow()
{
  cd "$SOURCE_DIR"

  echo -e "\n***** Downloading TensorFlow *****"
  git clone https://github.com/tensorflow/tensorflow.git
  cd "$TENSORFLOW_SRC"

  git checkout "$TENSORFLOW_VERSION"
  echo -e "\n***** TensorFlow downloaded *****"
}

build_tflite_cpuinfo()
{
  cd "$TFLITE_BUILD_TARGET"/cpuinfo
  cmake .
  make
  cp *.a ../_deps/cpuinfo-build
}

build_tflite()
{
  mkdir -p "$TFLITE_BUILD_TARGET"
  cd "$TFLITE_BUILD_TARGET"

  local target_arch_cmd="" # default is native, no command needed
  local cmake_flags=""

  case "$TARGET_ARCH" in
    "aarch64")
      cmake_system="Linux"
      if [ "$os_darwin" -eq 1 ]; then
        cmake_system="Darwin"
      fi
      cmake_flags+="$AARCH64_COMPILER_FLAGS"
      target_arch_cmd="-DCMAKE_SYSTEM_PROCESSOR=aarch64 \
                       -DCMAKE_SYSTEM_NAME=$cmake_system "

      if [ "$NATIVE_BUILD" -eq 0 ]; then
        cmake_flags+="ARMCC_FLAGS='-funsafe-math-optimizations' "
      fi
      ;;
    "android64")
      cmake_flags+="$ANDROID64_COMPILER_FLAGS"
      if [ "$NATIVE_BUILD" -eq 0 ]; then
        target_arch_cmd="-DCMAKE_TOOLCHAIN_FILE=$NDK_SRC/build/cmake/android.toolchain.cmake \
                         -DANDROID_ABI=$ANDROID_ARM_ARCH \
                         -DANDROID_PLATFORM=$ANDROID_API_VERSION"
      fi
      ;;
  esac

  echo -e "\n***** Building TF Lite for $TARGET_ARCH *****"

  # Cleanup any previous cmake files, except actual builds which we keep
  find . -mindepth 1 -name "*_build" -prune -o -exec rm -rf {} +

  eval "$cmake_flags" \
  cmake -DTFLITE_ENABLE_XNNPACK=OFF \
        -DFLATBUFFERS_BUILD_FLATC=OFF \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        "$target_arch_cmd" \
        "$TFLITE_SRC"
  cmake --build . -j "$NUM_THREADS"

  if [ "$os_darwin" -eq 1 ]; then
    # Workaround undefined link error for this platform
    build_tflite_cpuinfo
  fi

  echo -e "\n***** Built TF Lite for $TARGET_ARCH *****"
}

generate_tflite_schema()
{
  echo -e "\n***** Generating TF Lite Schema *****"
  mkdir -p "$TFLITE_BUILD_ROOT"
  cd "$TFLITE_BUILD_ROOT"

  cp "$SCHEMA_SRC" .

  $FLATC -c --gen-object-api --reflect-types --reflect-names schema.fbs

  echo -e "\n***** Generated TF Lite Schema *****"
}

download_onnx()
{
  download_and_extract \
    "ONNX" \
    "https://github.com/onnx/onnx/releases/download/v$ONNX_VERSION/onnx-$ONNX_VERSION.tar.gz" \
    "onnx-$ONNX_VERSION.tar.gz"
}

generate_onnx_sources()
{
  mkdir -p "$ONNX_BUILD_TARGET"
  cd "$ONNX_SRC"

  echo -e "\n***** Generating ONNX sources for $TARGET_ARCH *****"

  export LD_LIBRARY_PATH="$PROTOBUF_BUILD_HOST"/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

  eval "$PROTOCOL_COMPILER_HOST" onnx/onnx.proto \
  --proto_path=. \
  --proto_path="$ONNX_SRC" \
  --proto_path="$PROTOBUF_BUILD_HOST"/include \
  --cpp_out "$ONNX_BUILD_TARGET"

  echo -e "\n***** Generated ONNX sources for $TARGET_ARCH *****"
}

usage()
{
  cat <<EOF
setup-armnn.sh - Download and build Arm NN dependencies in the current directory (ROOT_DIR)
setup-armnn.sh [OPTION]...
  --tflite-classic-delegate
    setup dependencies for the existing Arm NN TF Lite Delegate
  --tflite-opaque-delegate
    setup dependencies for the new Opaque Delegate
  --tflite-parser
    setup dependencies for the Arm NN TF Lite Parser
  --onnx-parser
    setup dependencies for the Arm NN ONNX parser
  --all
    setup dependencies for all Arm NN components listed above
  --target-arch=[aarch64|android64|x86_64]
    specify a target architecture (mandatory)
  --num-threads=<INTEGER>
    specify number of threads/cores to build dependencies with (optional: defaults to number of online CPU cores on host)
  -h, --help
    print brief usage information and exit
  -x
    enable shell tracing in this script

At least one dependency flag (e.g. --tflite-classic-delegate) must be provided or else provide --all to setup all dependencies.
Directories called "source" and "build" will be generated in the current directory (ROOT_DIR) from which this script is called.
It's recommended to call this script in a directory outside of this Arm NN source repo, to avoid nested repositories.

Examples:
Setup for aarch64 with all Arm NN dependencies:
    <PATH_TO>/setup-armnn.sh --target-arch=aarch64 --all
Setup for aarch64 with the existing TF Lite Delegate and TF Lite Parser dependencies only:
    <PATH_TO>/setup-armnn.sh --target-arch=aarch64 --tflite-classic-delegate --tflite-parser
EOF
}

# This will catch in validation.sh if not set
target_arch=""

# Default flag values
flag_tflite_classic_delegate=0
flag_tflite_opaque_delegate=0
flag_tflite_parser=0
flag_onnx_parser=0

# If --num-threads is not set, the default NUM_THREADS value in common.sh will be used
num_threads=0

name=$(basename "$0")

# If no options provided, show help
if [ $# -eq 0 ]; then
  usage
  exit 1
fi

args=$($osgetopt -ohx -l tflite-classic-delegate,tflite-opaque-delegate,tflite-parser,onnx-parser,all,target-arch:,num-threads:,help -n "$name"   -- "$@")
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

  --tflite-classic-delegate)
    flag_tflite_classic_delegate=1
    ;;

  --tflite-opaque-delegate)
    flag_tflite_opaque_delegate=1
    ;;

  --onnx-parser)
    flag_onnx_parser=1
    ;;

  --all)
    flag_tflite_classic_delegate=1
    flag_tflite_opaque_delegate=1
    flag_tflite_parser=1
    flag_onnx_parser=1
    ;;

  --target-arch)
    opt_prev=target_arch
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

echo -e "\nINFO: Displaying configuration information before execution of $name"
echo "            target-arch: $TARGET_ARCH"
echo "              host-arch: $HOST_ARCH"
echo "tflite-classic-delegate: $flag_tflite_classic_delegate"
echo "tflite-opaque-delegate : $flag_tflite_opaque_delegate"
echo "          tflite-parser: $flag_tflite_parser"
echo "            onnx-parser: $flag_onnx_parser"
echo "            num-threads: $NUM_THREADS"
echo "         root directory: $ROOT_DIR"
echo "       source directory: $SOURCE_DIR"
echo "        build directory: $BUILD_DIR"

if check_if_repository .; then
  echo -e "\n***** WARNING: Running script inside a git repository. To avoid nested repos, call this script from outside of this repo. *****"
fi

echo -e "\nScript execution will begin in 10 seconds..."

sleep 10

mkdir -p "$SOURCE_DIR"
mkdir -p "$BUILD_DIR"

if [ "$TARGET_ARCH" == "android64" ]; then
  download_androidndk
fi

if [ "$flag_onnx_parser" -eq 1 ] || [ "$flag_tflite_classic_delegate" -eq 1 ] || [ "$flag_tflite_opaque_delegate" -eq 1 ] || [ "$flag_tflite_parser" -eq 1 ]; then
  download_flatbuffers

  # Host build
  build_flatbuffers 1

  # Target build for cross compile
  if [ "$NATIVE_BUILD" -eq 0 ]; then
    build_flatbuffers 0
  fi

  if [ "$flag_tflite_classic_delegate" -eq 1 ] || [ "$flag_tflite_opaque_delegate" -eq 1 ] || [ "$flag_tflite_parser" -eq 1 ]; then
    download_tensorflow
  fi
fi

if [ "$flag_tflite_parser" -eq 1 ]; then
  generate_tflite_schema
fi

if [ "$flag_tflite_classic_delegate" -eq 1 ] || [ "$flag_tflite_opaque_delegate" -eq 1 ]; then
  build_tflite
fi

if [ "$flag_onnx_parser" -eq 1 ]; then
  download_protobuf

  # Host build
  build_protobuf 1

  # Target build for cross compile
  if [ "$NATIVE_BUILD" -eq 0 ]; then
    build_protobuf 0
  fi

  download_onnx
  generate_onnx_sources
fi

echo -e "\n***** Arm NN setup complete. Now build with build-armnn.sh. *****\n"

exit 0
