#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMNNTOSACOMMON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/tosaCommon)
    list(APPEND armnnLibraries armnnTosaBackend)
    list(APPEND armnnLibraries armnnTosaBackendOperators)
    list(APPEND armnnUnitTestLibraries armnnTosaBackendUnitTests)
endif()
