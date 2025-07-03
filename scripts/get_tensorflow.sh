#!/bin/bash
#
# Copyright © 2022-2025 Arm Ltd and Contributors. All rights reserved.
#
# SPDX-License-Identifier: MIT
#

CMD=$( basename "$0" )

# Revision or tag that Arm NN has been tested with:
DEFAULT_TENSORFLOW_REVISION=v2.19.0 # tag v2.19.0

Usage() {
    echo "Gets the revision or tag of TensorFlow that this version of Arm NN has been" 
    echo "tested with."
    echo
    echo "Usage: $CMD Gets the default TensorFlow revision/tag ($DEFAULT_TENSORFLOW_REVISION)"
    echo "Usage: $CMD -s <TENSORFLOW_SHA>"
    echo "Usage: $CMD -p (Print current default revision/tag)"
    exit 0
}

PrintDefaultTensorFlowSha() {
  echo $DEFAULT_TENSORFLOW_REVISION
  exit 0;
}

function AssertZeroExitCode {
  EXITCODE=$?
  if [ $EXITCODE -ne 0 ]; then
    echo "$1"
    echo "+++ Command exited with code $EXITCODE. Please fix the above errors and re-run"
    exit 1
  fi
}

# Revision or tag to check out
TENSORFLOW_REVISION=$DEFAULT_TENSORFLOW_REVISION

# process the options given
while getopts "s:ph" opt; do
  case "$opt" in
    s) TENSORFLOW_REVISION="$OPTARG";;
    p) PrintDefaultTensorFlowSha;;
    h) Usage;;
  esac
done
shift $((OPTIND - 1))

#
# This script is designed to be called from anywhere
# so it will resolve where to checkout out TensorFlow
# relative to its own location in armnn/scripts
#
SRC="${BASH_SOURCE[0]}"
# resolve $SRC until it is no longer a symlink
while [ -h "$SRC" ]; do
  DIR="$( cd -P "$( dirname "$SRC" )" >/dev/null && pwd )"
  SRC="$(readlink "$SRC")"
  # if $SRC was a relative symlink, we need to resolve it
  # relative to the path where the symlink file originally was
  [[ $SRC != /* ]] && SRC="$DIR/$SRC"
done
DIR="$( cd -P "$( dirname "$SRC" )" >/dev/null && pwd )"
pushd "${DIR}" > /dev/null
cd ../.. || exit

# Clone TensorFlow if we don't already have a directory
if [ ! -d tensorflow ]; then
  echo "Cloning TensorFlow"
  # Attempt to clone Tensorflow, wait 60 second between attempts. Max 5 tries
  n=0
  until [ $n -ge 5 ]
  do
    git clone https://github.com/tensorflow/tensorflow.git && break
    n=$[$n+1]
    sleep 60
  done
  AssertZeroExitCode "Cloning TensorFlow failed"
fi
pushd tensorflow > /dev/null

# Checkout the TensorFlow revision
echo "Checking out ${TENSORFLOW_REVISION}"
git fetch && git checkout "${TENSORFLOW_REVISION}"
AssertZeroExitCode "Fetching and checking out ${TENSORFLOW_REVISION} failed"
# If the target tensorflow revision includes a branch we also need to do a pull.
# This generally occurs with a release branch.
if [[ "${TENSORFLOW_REVISION}" == *"branches"* ]]; then
    git pull
    AssertZeroExitCode "TensorFlow reference includes a branch but git pull failed."
fi

popd > /dev/null # out of tensorflow
popd > /dev/null # back to wherever we were when called
# Make sure the SHA of the revision that was checked out is the last line
# of output from the script... just in case we ever need it.
echo "$TENSORFLOW_REVISION"
exit 0
