#
# Copyright © 2023, 2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

include(FindPackageHandleStandardArgs)
unset(TFLITEABSL_FOUND)

find_path(TfLite_ABSL_SYNC_HEADERS
        NAMES
            absl
        PATHS
            ${TFLITE_LIB_ROOT}/abseil-cpp
            NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

# First look for the static version of tensorflow lite
find_library(TfLite_LIB NAMES "libtensorflow-lite.a" PATHS ${TFLITE_LIB_ROOT} ${TFLITE_LIB_ROOT}/tensorflow/lite NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH )

# If the static library was found, gather extra absl libraries for opaque delegate
if (TfLite_LIB MATCHES .a$)
    find_library(TfLite_abseil_base_LIB "libabsl_base.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/base NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH )
    find_library(TfLite_abseil_log_severity_LIB "libabsl_log_severity.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/base NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_spinlock_wait_LIB "libabsl_spinlock_wait.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/base NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_malloc_internal_LIB "libabsl_malloc_internal.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/base NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_raw_logging_internal_LIB "libabsl_raw_logging_internal.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/base NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_stacktrace_LIB "libabsl_stacktrace.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/debugging NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_debugging_internal_LIB "libabsl_debugging_internal.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/debugging NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_symbolize_LIB "libabsl_symbolize.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/debugging NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_demangle_internal_LIB "libabsl_demangle_internal.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/debugging NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_time_LIB "libabsl_time.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/time NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_time_zone_LIB "libabsl_time_zone.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/time NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
    find_library(TfLite_abseil_int128_LIB "libabsl_int128.a" PATHS
                ${TFLITE_LIB_ROOT}/_deps/abseil-cpp-build/absl/numeric NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)

    ## Set TFLITEABSL_FOUND
    find_package_handle_standard_args(TfLiteAbsl DEFAULT_MSG TfLite_ABSL_SYNC_HEADERS TfLite_abseil_base_LIB
                                      TfLite_abseil_int128_LIB TfLite_abseil_spinlock_wait_LIB
                                      TfLite_abseil_raw_logging_internal_LIB TfLite_abseil_malloc_internal_LIB
                                      TfLite_abseil_symbolize_LIB  TfLite_abseil_stacktrace_LIB
                                      TfLite_abseil_demangle_internal_LIB TfLite_abseil_debugging_internal_LIB
                                      TfLite_abseil_time_LIB TfLite_abseil_time_zone_LIB)

    ## Set external variables for usage in CMakeLists.txt
    if(TFLITEABSL_FOUND)
        set(TfLite_ABSL_SYNC_HEADERS ${TfLite_ABSL_SYNC_HEADERS})
        set(TfLite_Extra_Absl_LIB ${TfLite_abseil_base_LIB} ${TfLite_abseil_int128_LIB} ${TfLite_abseil_spinlock_wait_LIB}
                                  ${TfLite_abseil_raw_logging_internal_LIB} ${TfLite_abseil_malloc_internal_LIB}
                                  ${TfLite_abseil_symbolize_LIB}  ${TfLite_abseil_stacktrace_LIB}
                                  ${TfLite_abseil_demangle_internal_LIB} ${TfLite_abseil_debugging_internal_LIB}
                                  ${TfLite_abseil_time_LIB} ${TfLite_abseil_time_zone_LIB})
    endif()
endif()
