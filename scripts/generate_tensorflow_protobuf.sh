#!/bin/sh
#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

THIS_SCRIPT=$0
OUTPUT_DIR=$1
PROTOBUF_INSTALL_DIR=$2

usage()
{
  echo
  echo "Usage: ${THIS_SCRIPT} <OUTPUT_DIR> [PROTOBUF_INSTALL_DIR]"
  echo
  echo "  <OUTPUT_DIR> is the location where the generated files will be placed"
  echo "  [PROTOBUF_INSTALL_DIR] the location of the protobuf installation"
  echo
}

if [ "x$OUTPUT_DIR" = "x" ]
then
  usage
  exit 1
fi

mkdir -p ${OUTPUT_DIR}
ERR=$?
if [ $ERR -ne 0 ]
then
  echo
  echo "Cannot create output dir: ${OUTPUT_DIR}"
  echo "mkdir returned: $ERR"
  echo
  usage
  exit 1
fi


if [ "x${PROTOBUF_INSTALL_DIR}" = "x" ]
then
  PROTOBUF_INSTALL_DIR=/usr/local
fi

if [ ! -x "${PROTOBUF_INSTALL_DIR}/bin/protoc" ]
then
  echo
  echo "No usable protocol buffer (protoc) compiler found in ${PROTOBUF_INSTALL_DIR}/bin/"
  echo "You can specify the location of the protobuf installation as the second"
  echo "argument of ${THIS_SCRIPT}."
  usage
  exit 1
fi

OLD_LD_LIBRARY_PATH=$LD_LIBRARY_PATH
#hardware_types.proto and autotuning.proto not required
find tensorflow -type f -name '*.proto' | grep -v autotuning | grep -v hardware_types | while read i; do
  LD_LIBRARY_PATH=$OLD_LD_LIBRARY_PATH:${PROTOBUF_INSTALL_DIR}/lib $PROTOBUF_INSTALL_DIR/bin/protoc $i \
    --proto_path=. \
    --proto_path=${PROTOBUF_INSTALL_DIR}/include \
    --cpp_out $OUTPUT_DIR
  EXIT_CODE=$?
  if [ $EXIT_CODE -ne 0 ]; then
    echo "Failed to make proto files"
    exit 1
  fi
done