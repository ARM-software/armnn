#!/bin/bash
#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Script which copies a file or directory from the /home/arm-user/ directory in Docker to the host machine
# This script creates a directory called 'docker_output' in the current directory and places the copied contents there
# Takes two arguments:
# 1. Name of created Docker image i.e. "--tag <name:tag>" provided at 'docker build' stage (tag is optional in image naming)
# 2. Relative path to file or directory to copy from the Docker /home/arm-user/ directory
#
# Examples:
# 1. Copy the tarball of the aarch64 build from the /home/arm-user/ directory
#    ./scripts/docker-copy-to-host.sh armnn_image armnn_aarch64_build.tar.gz
# 2. Copy the unarchived Arm NN build
#    ./scripts/docker-copy-to-host.sh armnn_image build/armnn
# 3. Copy the unarchived ACL build
#    ./scripts/docker-copy-to-host.sh armnn_image build/acl

set -o nounset  # Catch references to undefined variables.
set -o pipefail # Catch non zero exit codes within pipelines.
set -o errexit  # Catch and propagate non zero exit codes.

image_name="$1"
file_path="$2"

name=$(basename "$0")

echo "***** $name: Copying file(s) from path /home/arm-user/$file_path inside Docker image '$image_name' to host *****"

echo -e "\n***** Creating directory docker_output on host *****"
mkdir -p docker_output

# Cleanup old 'armnn_temp' container in case a previous run of this script was not successful
docker rm --force armnn_temp 2> /dev/null

echo -e "\n***** Creating temporary Docker container named armnn_temp using Docker image '$image_name' *****"
docker create --interactive --tty --name armnn_temp "$image_name" bash > /dev/null

echo -e "\n***** Running Docker command: docker cp armnn_temp:/home/arm-user/$file_path ./docker_output *****"
docker cp armnn_temp:/home/arm-user/"$file_path" ./docker_output > /dev/null

echo -e "\n***** Successfully copied file(s) to host in directory docker_output *****"

# Remove temporary docker container 'armnn_temp'
docker rm --force armnn_temp > /dev/null

echo -e "\n***** Deleted temporary Docker container armnn_temp *****"