#!/bin/bash
#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  TARGET="$(readlink "$SOURCE")"
  if [[ $TARGET == /* ]]; then
    # "SOURCE '$SOURCE' is an absolute symlink to '$TARGET'"
    SOURCE="$TARGET"
  else
    DIR="$( dirname "$SOURCE" )"
    # "SOURCE '$SOURCE' is a relative symlink to '$TARGET' (relative to '$DIR')"
    SOURCE="$DIR/$TARGET" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
  fi
done
RDIR="$( dirname "$SOURCE" )"
DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null 2>&1 && pwd )"

CMD=$( basename $0 )

usage() {
  echo "Usage: $CMD [options]"
  echo "Options:        -t(type) <Debug or Release>"
  echo "                -c(lean) build"
  echo "                -s(tatic libraries) <1 or 0>"
  echo "                -w(indows) build"
  exit 1
}
# defaults
TYPE=Release
CLEAN=0
STATIC=0
WINDOWS=0

# Parse the command line
while getopts "whct:s:" opt; do
  case "$opt" in
    h|\?) usage;;
    t) TYPE=$OPTARG;;
    c) CLEAN=1;;
    s) STATIC=$OPTARG;;
    w) WINDOWS=1;;
  esac
done
shift $((OPTIND - 1))

if [ $CLEAN == 1 ]; then
    echo "removing ${DIR}/build"
    rm -rf ${DIR}/build
fi

BUILD_DIR="build"
[ -d build ] || mkdir build
echo $WINDOWS
if [ "$WINDOWS" -eq "1" ]; then
    echo "doing windows"
    cd $BUILD_DIR
    [ -d windows ] || mkdir windows
    BUILD_DIR=$BUILD_DIR/windows
    cd $DIR
fi
# lower case TYPE in a posix compliant manner
LC_TYPE=$(echo "$TYPE" | tr '[:upper:]' '[:lower:]')
if [ ${LC_TYPE} == "debug" ]; then
    DEBUGDIR=($DIR/$BUILD_DIR/debug)
    [ -d $DEBUGDIR ] || (cd ${BUILD_DIR} && mkdir debug && cd ..)
    BUILD_DIR=$DEBUGDIR
else
    RELEASEDIR=($DIR/$BUILD_DIR/release)
    [ -d $RELEASEDIR ] || (cd ${BUILD_DIR} && mkdir release && cd ..)
    BUILD_DIR=$RELEASEDIR
fi

echo "Build Directory: ${BUILD_DIR}"

CMAKE=cmake
CMARGS="-DCMAKE_BUILD_TYPE=$TYPE \
        -DBUILD_STATIC_PIPE_LIBS=$STATIC \
        -DBUILD_PIPE_ONLY=1"
if [ "$WINDOWS" -eq "1" ]; then
    CMARGS="$CMARGS \
           -DCMAKE_TOOLCHAIN_FILE=${DIR}/toolchain-x86-ubuntu-mingw64.cmake"
fi
MAKE=make

cd ${BUILD_DIR}
pwd
( eval $CMAKE $CMARGS $DIR && eval ${MAKE} $MAKEFLAGS )
