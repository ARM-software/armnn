#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTENEON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/neon)
    list(APPEND armnnLibraries armnnNeonBackend armnnNeonBackendWorkloads)
    list(APPEND armnnUnitTestLibraries armnnNeonBackendUnitTests)
else()
    message("NEON backend is disabled")
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/neon)
    list(APPEND armnnLibraries armnnNeonBackend)
endif()
