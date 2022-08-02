#!/bin/bash
#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

#
# Script which uses the reuse license compliance tool: https://reuse.software/
# to do the following:
#  (a) check armnn for compliance
#  (b) generate an SPDX file
#  (c) insert into the SPDX file before the individual files section, SPDX files for
#      all the third-party header only source libraries used by Arm NN
#      to create a comprehensive LICENSE.spdx file for the armnn source code distribution
#
# Note to run correctly the script has to be run from the armnn root directory like so:
#
# ./scripts/generate_spdx.sh
#

# Check if the parent directory name is armnn
# Get the name of the current directory
result=${PWD##*/}          # to assign to a variable
result=${result:-/}        # to correct for the case where PWD=/

if [[ $result != "armnn" ]]
then
    echo "not running from armnn directory"
    exit -2
fi

# Check that the third-party subdirectory exists
if [ ! -d "third-party" ]; then
    echo "third-party directory does not exist."
    exit -3
fi

# Check that armnn is compliant with version 3.0 of the REUSE Specification
reuse lint
if [[ $? -ne 0 ]]
then
    echo "    "
    echo "please make armnn compliant with version 3.0 of the REUSE Specification before re-running"
    exit -4
fi

# generate the SPDX file for the overall armnn package
reuse spdx > LICENSE.spdx
if [[ $? -ne 0 ]]
then
    echo "generation of LICENSE.spdx file failed"
    exit -5
else
    echo "  "
    echo "LICENSE.spdx file generated"
    echo "  "
fi

# Add the license info for the third-party packages
# NOTE: they will be added before the first individual file entry
#       which currently is './Android.bp'

# insert header comment before the line: FileName: ./Android.bp
sed -i '/FileName: \.\/Android.bp/i \
##### Source dependencies \
# Header only libraries from the armnn source repository third-party folder \
# NOTE: fmt has a small .cc file that needs to be compiled in order to work hence the libfmt.a below in the static dependencies \
  ' LICENSE.spdx

# iterate over the LICENSE.spdx files in the third-party directory and
# put their contents into the top level LICENSE.spdx file
# before the line: FileName: ./Android.bp

for i in ./third-party/**/LICENSE.spdx;
do
    echo "inserting license $i"
    sed -i "/FileName: \.\/Android.bp/e cat $i" LICENSE.spdx
    sed -i '/FileName: \.\/Android.bp/i \
  ' LICENSE.spdx
done

# Mark the start of the individual files section of the file with a comment
sed -i '/FileName: \.\/Android.bp/i \
##### Individual Files \
   ' LICENSE.spdx
