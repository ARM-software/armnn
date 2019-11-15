#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/neon)
list(APPEND armnnLibraries armnnNeonBackend)

if(ARMCOMPUTENEON)
    list(APPEND armnnLibraries armnnNeonBackendWorkloads)
    list(APPEND armnnUnitTestLibraries armnnNeonBackendUnitTests)
else()
    message(STATUS "NEON backend is disabled")
endif()
