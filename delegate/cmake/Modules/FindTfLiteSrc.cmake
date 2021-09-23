#
# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

include(FindPackageHandleStandardArgs)
unset(TFLITE_SRC_FOUND)

find_path(TfLite_INCLUDE_DIR
        NAMES
            tensorflow/lite
            third_party
        HINTS
            ${TENSORFLOW_ROOT})

find_path(TfLite_Schema_INCLUDE_PATH
            schema_generated.h
        HINTS
            ${TENSORFLOW_ROOT}/tensorflow/lite/schema)

## Set TFLITE_FOUND
find_package_handle_standard_args(TfLiteSrc DEFAULT_MSG TfLite_INCLUDE_DIR TfLite_Schema_INCLUDE_PATH)

## Set external variables for usage in CMakeLists.txt
if(TFLITE_SRC_FOUND)
    set(TfLite_INCLUDE_DIR ${TfLite_INCLUDE_DIR})
    set(TfLite_Schema_INCLUDE_PATH ${TfLite_Schema_INCLUDE_PATH})
endif()
