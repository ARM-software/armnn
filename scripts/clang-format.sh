#!/bin/bash
#
# Copyright © 2021-2023 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

#
# Usage: clang-format.sh [path to file or directory]
#
# Formats all git-tracked files in the repository according to clang-format rules.
#
# Takes an optional parameter to limit the scope to a specific directory or file.
#
# The clang-format rules can be found in the top level directory: .clang-format
# To prevent clang-format on a section of code, use the following:
#       // clang-format off
#       my code that i dont want to be clang-formatted goes here
#       // clang-format on
#

# Check that clang-format is available.
CLANG_FORMAT=`command -v clang-format`
if [ -z $CLANG_FORMAT ]; then
  # Maybe path isn't set. Try a well known location.
  CLANG_FORMAT=/usr/bin/clang-format
  if [ ! -x "$CLANG_FORMAT" ]; then
    echo "Unable to locate clang-format. Try installing it: sudo apt-get install clang-format"
  fi
fi

# Find all hpp and cpp files excluding some directories.
files=$(git ls-files ${1} | egrep -v '^(build-tool|docker|docs|generated|third-party)' | egrep '\.[ch](pp)?$')

num_files_changed=0

for file in $files; do
    echo "[CLANG-FORMAT] ${file}"
    # Run clang-format in-place to update the file
    $CLANG_FORMAT -style=file ${file} -i
    if [ $? -ne 0 ]; then
      echo "Error: Execution of $CLANG_FORMAT failed."
      exit -1
    else
      ((num_files_changed++))
    fi
done

echo "${num_files_changed} file(s) changed"
exit ${num_files_changed}
