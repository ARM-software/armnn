#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(Tensorflow_INCLUDE_DIR
        NAMES
            tensorflow/core
            tensorflow/cc
            third_party
        HINTS
            ${TENSORFLOW_ROOT})

find_library(Tensorflow_LIB
        NAMES
            tensorflow_all
        HINTS
            ${TENSORFLOW_LIB_DIR})

## Set TENSORFLOW_FOUND
find_package_handle_standard_args(Tensorflow DEFAULT_MSG Tensorflow_INCLUDE_DIR Tensorflow_LIB)

## Set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(Tensorflow_LIB ${Tensorflow_LIB})
    set(Tensorflow_INCLUDE_DIRS ${Tensorflow_INCLUDE_DIR})
endif()