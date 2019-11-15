#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/cl)
list(APPEND armnnLibraries armnnClBackend)

if(ARMCOMPUTECL)
    list(APPEND armnnLibraries armnnClBackendWorkloads)
    list(APPEND armnnUnitTestLibraries armnnClBackendUnitTests)
else()
    message(STATUS "CL backend is disabled")
endif()
