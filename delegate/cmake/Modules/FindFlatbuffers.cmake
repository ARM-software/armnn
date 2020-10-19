#
# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

include(FindPackageHandleStandardArgs)
unset(FLATBUFFERS_FOUND)

find_path(Flatbuffers_INCLUDE_DIR
        flatbuffers/flatbuffers.h
        HINTS
            ${FLATBUFFERS_ROOT}/include
            /usr/local/include
            /usr/include)

find_library(Flatbuffers_LIB
        NAMES
            libflatbuffers.a
            flatbuffers
        HINTS
            ${FLATBUFFERS_ROOT}/lib
            /usr/local/lib
            /usr/lib)

## Set FLATBUFFERS_FOUND
find_package_handle_standard_args(Flatbuffers DEFAULT_MSG Flatbuffers_INCLUDE_DIR Flatbuffers_LIB)

## Set external variables for usage in CMakeLists.txt
if(FLATBUFFERS_FOUND)
    set(Flatbuffers_LIB ${Flatbuffers_LIB})
    set(Flatbuffers_INCLUDE_DIR ${Flatbuffers_INCLUDE_DIR})
endif()