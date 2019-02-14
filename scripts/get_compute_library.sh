#!/bin/bash
#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

CMD=$( basename $0 )

usage() {
  echo "Usage: $CMD -g <GITHUB_USERNAME>"
  exit 1
}

function AssertZeroExitCode {
  EXITCODE=$?
  if [ $EXITCODE -ne 0 ]; then
    echo "$1"
    echo "+++ Command exited with code $EXITCODE. Please fix the above errors and re-run"
    exit 1
  fi
}

# process the options given
while getopts "g:h" opt; do
  case "$opt" in
    g) GITHUB_USERNAME="$OPTARG";;
    h|\?) usage;;
  esac
done
shift $((OPTIND - 1))

#
# This script is designed to be called from anywhere
# so it will resolve where to checkout out the clframework
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
pushd ${DIR} > /dev/null
cd ../..
if [ -z "$USERNAME" ]; then
    USERNAME=$USER
fi
if [ -z "$GITHUB_USERNAME" ]; then
    GITHUB_USERNAME=$USERNAME
    echo "setting GITHUB_USERNAME: ${GITHUB_USERNAME} use -g command line option to change"
fi

if [ ! -d clframework ]; then
echo "+++ Cloning clframework"
  git clone https://review.mlplatform.org/ml/ComputeLibrary clframework
  AssertZeroExitCode "Cloning CL Framework failed"
fi
pushd clframework > /dev/null

# Use the latest pinned version of the CL framework

# For pinnning to a ref use this:
# CLFRAMEWORKREVISION="branches/arm_compute_18_11" # Release 18.11
# git fetch  https://review.mlplatform.org/ml/ComputeLibrary $CLFRAMEWORKREVISION && git checkout FETCH_HEAD

# For pinning to a revision use this:
CLFRAMEWORKREVISION="6eb90d7d8ef73decf6e3973a89c2b2badd0b9635" # Master towards 19.02
git fetch https://review.mlplatform.org/ml/ComputeLibrary && git checkout ${CLFRAMEWORKREVISION}
AssertZeroExitCode

# Set commit hook so we can submit reviews to gerrit
(curl -Lo `git rev-parse --git-dir`/hooks/commit-msg https://review.mlplatform.org/tools/hooks/commit-msg; chmod +x `git rev-parse --git-dir`/hooks/commit-msg)
AssertZeroExitCode

popd > /dev/null # out of clframework
popd > /dev/null # back to wherever we were when called
exit 0
