#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

include(FindPackageHandleStandardArgs)
unset(TFLITE_FOUND)

#
# NOTE: this module is used to find the tensorflow lite binary libraries only
#       the FindTfLiteSrc.cmake module is used to find the tensorflow lite include directory.
#       This is to allow components like the Tensorflow lite parser that have a source dependency
#       on tensorflow lite headers but no need to link to the binary libraries to use only the sources
#       and not have an artificial dependency on the libraries.
#

# First look for the static version of tensorflow lite
find_library(TfLite_LIB NAMES "libtensorflow-lite.a" HINTS ${TFLITE_LIB_ROOT} ${TFLITE_LIB_ROOT}/tensorflow/lite)
# If not found then, look for the dynamic library of tensorflow lite
find_library(TfLite_LIB NAMES "libtensorflow_lite_all.so" "libtensorflowlite.so" HINTS ${TFLITE_LIB_ROOT} ${TFLITE_LIB_ROOT}/tensorflow/lite)

# If the static library was found, gather all of its dependencies
if (TfLite_LIB MATCHES .a$)
    message("-- Static tensorflow lite library found, using for ArmNN build")
    find_library(TfLite_abseilstrings_LIB "libabsl_strings.a"
                 PATH ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/strings)
    find_library(TfLite_farmhash_LIB "libfarmhash.a"
                 PATH ${TFLITE_LIB_ROOT}/_deps/farmhash-build)
    find_library(TfLite_fftsg_LIB "libfft2d_fftsg.a"
                 PATH ${TFLITE_LIB_ROOT}/_deps/fft2d-build)
    find_library(TfLite_fftsg2d_LIB "libfft2d_fftsg2d.a"
                 PATH ${TFLITE_LIB_ROOT}/_deps/fft2d-build)
    find_library(TfLite_ruy_LIB "libruy.a" PATH
                 ${TFLITE_LIB_ROOT}/_deps/ruy-build)
    find_library(TfLite_flatbuffers_LIB "libflatbuffers.a"
                 PATH ${TFLITE_LIB_ROOT}/_deps/flatbuffers-build)

    ## Set TFLITE_FOUND if all libraries are satisfied for static lib
    find_package_handle_standard_args(TfLite DEFAULT_MSG TfLite_LIB TfLite_abseilstrings_LIB TfLite_ruy_LIB TfLite_fftsg_LIB TfLite_fftsg2d_LIB TfLite_farmhash_LIB TfLite_flatbuffers_LIB)
    # Set external variables for usage in CMakeLists.txt
    if (TFLITE_FOUND)
        set(TfLite_LIB ${TfLite_LIB} ${TfLite_abseilstrings_LIB} ${TfLite_ruy_LIB} ${TfLite_fftsg_LIB} ${TfLite_fftsg2d_LIB} ${TfLite_farmhash_LIB} ${TfLite_flatbuffers_LIB})
    endif ()
elseif (TfLite_LIB MATCHES .so$)
    message("-- Dynamic tensorflow lite library found, using for ArmNN build")
    find_package_handle_standard_args(TfLite DEFAULT_MSG TfLite_LIB)
    ## Set external variables for usage in CMakeLists.txt
    if (TFLITE_FOUND)
        set(TfLite_LIB ${TfLite_LIB})
    endif ()
else()
    message(FATAL_ERROR "Could not find a tensorflow lite library to use")
endif()
