# Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
# Search for ArmNN built libraries in user-provided path first, then current repository, then system

if( USE_ARMNN_DELEGATE )
    set(ARMNN_LIB_NAMES "libarmnn.so"
        "libarmnnDelegate.so")
else()
    set(ARMNN_LIB_NAMES "libarmnn.so"
        "libarmnnTfLiteParser.so")
endif()

set(ARMNN_LIBS "")

get_filename_component(PARENT_DIR ${PROJECT_SOURCE_DIR} DIRECTORY)
get_filename_component(REPO_DIR ${PARENT_DIR} DIRECTORY)

foreach(armnn_lib ${ARMNN_LIB_NAMES})
    find_library(ARMNN_${armnn_lib}
        NAMES
            ${armnn_lib}
        HINTS
            ${ARMNN_LIB_DIR} ${REPO_DIR}
        PATHS
            ${ARMNN_LIB_DIR} ${REPO_DIR}
        PATH_SUFFIXES
            "lib"
            "lib64")
    if(ARMNN_${armnn_lib})
        message("Found library ${ARMNN_${armnn_lib}}")
        list(APPEND ARMNN_LIBS ${ARMNN_${armnn_lib}})
        get_filename_component(LIB_DIR ${ARMNN_${armnn_lib}} DIRECTORY)
        get_filename_component(LIB_PARENT_DIR ${LIB_DIR} DIRECTORY)
        if( USE_ARMNN_DELEGATE )
            set(ARMNN_INCLUDE_DIR ${LIB_PARENT_DIR}/include
                ${PARENT_DIR}/../delegate/include
                ${PARENT_DIR}/../delegate/src)
        else()
            set(ARMNN_INCLUDE_DIR ${LIB_PARENT_DIR}/include)
        endif()
    endif()
endforeach()

if(NOT ARMNN_LIBS)
    message(FATAL_ERROR "Could not find ArmNN libraries ${ARMNN_LIB_NAMES}")
endif()
