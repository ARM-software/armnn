#!/bin/bash
#
# Copyright © 2018-2023 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

CMD=$( basename "$0" )

# For pinning to a ref use this:
#DEFAULT_CLFRAMEWORKREVISION="branches/arm_compute_23_02" # Release 23.02
#
# For pinning to a revision use this:
DEFAULT_CLFRAMEWORKREVISION="467daef993fe29cc4319058200b7ad797398e4b0" #9454: Implement CL kernel for a native batched matmul Quantized - LHS transposed, RHS transposed

usage() {
  echo -e "get_compute_library.sh: Clones the Arm Compute Library (ACL) repo from the ML Platform server and checks out
  the pinned version of ACL based on the SHA string defined at the top of this script (DEFAULT_CLFRAMEWORKREVISION).
  If the ACL repo already exists, this script will skip cloning and only checkout the relevant SHA.
  The pinned ACL version is a known version that works correctly with the version of Arm NN being used. This pin is
  regularly updated by the Arm NN team on the Arm NN 'main' branch.
  During release periods, the ACL pin will point to an ACL release branch which exists on the ML Platform server.
  The repo directory will be named 'clframework' unless defined by the '-n' argument to this script.\n"
  echo "Usage: $CMD (Use the default clframework SHA)"
  echo "Usage: $CMD -s <CLFRAMEWORK_SHA>"
  echo "Usage: $CMD -p (Print current default clframework SHA)"
  echo "Usage: $CMD -n Name of the directory into which the ACL repo will be cloned, default is 'clframework'"
  exit 0
}

PrintDefaultClframeworkSha() {
  echo $DEFAULT_CLFRAMEWORKREVISION
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

# process the options given
while getopts "s:n:ph" opt; do
  case "$opt" in
    s) CLFRAMEWORK_SHA="$OPTARG";;
    p) PrintDefaultClframeworkSha;;
    n) ACL_REPO_NAME_OPTION="$OPTARG";;
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
pushd "${DIR}" > /dev/null
# shellcheck disable=SC2164
cd ../..

# Default ACL repo directory name is 'clframework'
# This can be overwritten by command line option '-n'
ACL_REPO_NAME="clframework"
if [ ! -z "$ACL_REPO_NAME_OPTION" ]; then
  ACL_REPO_NAME="$ACL_REPO_NAME_OPTION"
fi

if [ ! -d "$ACL_REPO_NAME" ]; then
  echo "Cloning CL Framework"
  git clone https://review.mlplatform.org/ml/ComputeLibrary "$ACL_REPO_NAME"
  AssertZeroExitCode "Cloning CL Framework failed"
fi
pushd "$ACL_REPO_NAME" > /dev/null

CLFRAMEWORKREVISION=$DEFAULT_CLFRAMEWORKREVISION
if [ ! -z "$CLFRAMEWORK_SHA" ]; then
  CLFRAMEWORKREVISION=$CLFRAMEWORK_SHA
fi

echo "git fetch && git fetch https://review.mlplatform.org/ml/ComputeLibrary && git checkout $CLFRAMEWORKREVISION"
git fetch && git fetch https://review.mlplatform.org/ml/ComputeLibrary && git checkout "${CLFRAMEWORKREVISION}"
AssertZeroExitCode "Fetching and checking out ${CLFRAMEWORKREVISION} failed"
# If the target ACL revision includes a branch we also need to do a pull.
# This generally occurs with a release branch.
if [[ "${CLFRAMEWORKREVISION}" == *"branches"* ]]; then
  git pull
  AssertZeroExitCode "ACL reference includes a branch but git pull failed."
fi

# Set commit hook so we can submit reviews to gerrit
# shellcheck disable=SC2006
(curl -Lo "$(git rev-parse --git-dir)"/hooks/commit-msg https://review.mlplatform.org/tools/hooks/commit-msg; chmod +x "$(git rev-parse --git-dir)"/hooks/commit-msg)
AssertZeroExitCode "Setting commit hooks failed"

popd > /dev/null # out of clframework / "$ACL_REPO_NAME"
popd > /dev/null # back to wherever we were when called
# Make sure the SHA of the revision that was checked out is the last line
# of output from the script... just in case we ever need it.
echo "$CLFRAMEWORKREVISION"
exit 0
