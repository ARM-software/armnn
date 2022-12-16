#!/bin/bash
#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Script which installs system-wide packages required by setup-armnn.sh and build-armnn.sh
# Downloads and builds CMake from source in the current directory from which this script is called
# CMake will be installed system-wide once this script has completed execution
# Requires sudo privileges

set -o nounset  # Catch references to undefined variables.
set -o pipefail # Catch non zero exit codes within pipelines.
set -o errexit  # Catch and propagate non zero exit codes.

# Host architecture e.g. x86_64, aarch64
HOST_ARCH=$(uname -m)

# Number of online cores on host
NUM_THREADS=$(getconf _NPROCESSORS_ONLN)

# CMake is downloaded and built in the current directory from which this script is called
ROOT_DIR=$(pwd)

# CMake
CMAKE_VERSION=3.19
CMAKE_VERSION_FULL=3.19.0
CMAKE_SRC="$ROOT_DIR"/cmake-"$CMAKE_VERSION_FULL"
CMAKE_BUILD="$ROOT_DIR"/cmake_build

download_cmake()
{
  cd "$ROOT_DIR"

  echo -e "\n***** Downloading CMake $CMAKE_VERSION *****"
  wget -O cmake-"$CMAKE_VERSION_FULL".tar.gz https://cmake.org/files/v"$CMAKE_VERSION"/cmake-"$CMAKE_VERSION_FULL".tar.gz

  echo -e "\n***** Extracting archive *****"
  tar -xzf cmake-"$CMAKE_VERSION_FULL".tar.gz

  echo -e "\n***** Removing archive *****"
  rm cmake-"$CMAKE_VERSION_FULL".tar.gz

  echo -e "\n***** CMake $CMAKE_VERSION Downloaded *****"
}

install_cmake()
{
  mkdir -p "$CMAKE_BUILD"
  cd "$CMAKE_BUILD"

  apt-get purge -y cmake

  echo -e "\n***** Building CMake $CMAKE_VERSION ***** "
  "$CMAKE_SRC"/bootstrap
  make
  make install -j "$NUM_THREADS"

  if [[ "$(cmake --version 2> /dev/null | grep "$CMAKE_VERSION" )" == *"$CMAKE_VERSION"* ]]; then
    echo -e "\n***** Built and Installed CMake $CMAKE_VERSION *****"
  else
    echo -e "\nERROR: CMake $CMAKE_VERSION not installed correctly after building from source"
    exit 1
  fi
}

install_apt_packages()
{
  apt-get update && apt-get install -y --no-install-recommends  \
      autoconf \
      automake \
      build-essential \
      curl \
      git \
      libssl-dev \
      libtool \
      make \
      scons \
      unzip \
      wget \
      xxd

  # Install cross compile toolchains if host is x86_64
  if [ "$HOST_ARCH" == "x86_64" ]; then
    apt-get update && apt-get install -y --no-install-recommends  \
        crossbuild-essential-arm64
  fi

  apt-get clean
  rm -rf /var/lib/apt/lists/*
}

name=$(basename "$0")

if [ ! "$(id -u)" -eq 0 ]; then
  echo -e "\nERROR: $name must be ran as root (i.e. sudo ./$name)"
  exit 1
fi

echo -e "\n***** $name: Installing system-wide packages required by setup-armnn.sh and build-armnn.sh *****"
echo -e "\nINFO: This script downloads and builds CMake from source in the current directory from which this script is called"
echo -e "\nINFO: CMake and other apt packages will be installed system-wide once this script has completed execution"
echo -e "\nScript execution will begin in 10 seconds..."

sleep 10

install_apt_packages

# Download, Build and Install CMake if not already present
if [[ "$(cmake --version 2> /dev/null | grep "$CMAKE_VERSION" )" == *"$CMAKE_VERSION"* ]]; then
  echo -e "\n***** CMake $CMAKE_VERSION already installed, skipping CMake install *****"
else
  download_cmake
  install_cmake
fi

echo -e "\n***** $name: Successfully installed system-wide packages required by setup-armnn.sh and build-armnn.sh *****\n"

exit 0