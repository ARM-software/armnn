#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMNNTOSAREF)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/tosaCommon)
    list(APPEND armnnLibraries armnnTosaBackend)
    list(APPEND armnnLibraries armnnTosaBackendOperators)
    list(APPEND armnnUnitTestLibraries armnnTosaBackendUnitTests)
endif()
