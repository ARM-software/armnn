#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

include(FindPackageHandleStandardArgs)
unset(TFLITE_FOUND)

find_path(TfLite_INCLUDE_DIR
        NAMES
            tensorflow/lite
            third_party
        HINTS
            ${TENSORFLOW_ROOT})

find_library(TfLite_LIB
        NAMES
            "libtensorflow_lite_all.so"
        HINTS
            ${TFLITE_LIB_ROOT})

find_path(TfLite_Schema_INCLUDE_PATH
            schema_generated.h
        HINTS
            ${TFLITE_LIB_ROOT}/tensorflow/lite/schema)

## Set TFLITE_FOUND
find_package_handle_standard_args(TfLite DEFAULT_MSG TfLite_INCLUDE_DIR TfLite_LIB TfLite_Schema_INCLUDE_PATH)

## Set external variables for usage in CMakeLists.txt
if(TFLITE_FOUND)
    set(TfLite_LIB ${TfLite_LIB})
    set(TfLite_INCLUDE_DIR ${TfLite_INCLUDE_DIR})
    set(TfLite_Schema_INCLUDE_PATH ${TfLite_Schema_INCLUDE_PATH})
endif()