#!/bin/bash
#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Common validation of command line arguments provided to setup-armnn.sh and build-armnn.sh

# shellcheck disable=SC2034,SC2154
# SC2034: false positives for variables appear unused - variables are used in setup-armnn.sh and build-armnn.sh
# SC2154: false positives for variables referenced but not assigned - variables are assigned in setup-armnn.sh and build-armnn.sh

set -o nounset  # Catch references to undefined variables.
set -o pipefail # Catch non zero exit codes within pipelines.
set -o errexit  # Catch and propagate non zero exit codes.

# Host and target architecture validation
if [ "$target_arch" == "" ]; then
  echo "$name: --target-arch is not set. Example usage: --target-arch=aarch64"
  exit 1
fi

if [ "$target_arch" != "aarch64" ] && [ "$target_arch" != "x86_64" ]; then
  echo "$name: --target-arch is not valid. Valid options are: aarch64, x86_64"
  exit 1
fi

if [ "$HOST_ARCH" == "aarch64" ]; then
  if [ "$target_arch" != "aarch64" ]; then
    echo "$name: aarch64 is the only supported --target_arch when host is aarch64"
    exit 1
  fi
fi

# Validation of chosen Arm NN dependencies
if [ "$flag_tflite_delegate" -eq 0 ] && [ "$flag_tflite_parser" -eq 0 ] && [ "$flag_onnx_parser" -eq 0 ]; then
  echo "$name: at least one of flags --tflite-delegate, --tflite-parser or --onnx-parser must be set (or --all)."
  exit 1
fi

# If --num-threads is set, overwrite default NUM_THREADS with user-defined value
if [ ! "$num_threads" -eq 0 ]; then
  NUM_THREADS="$num_threads"
fi