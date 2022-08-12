#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/tosaReference)
list(APPEND armnnLibraries armnnTosaRefBackend)

if(ARMNNTOSAREF)
    list(APPEND armnnLibraries armnnTosaRefBackendWorkloads)
    list(APPEND armnnUnitTestLibraries armnnTosaRefBackendUnitTests)
else()
    message(STATUS "TOSA Reference backend is disabled")
endif()
