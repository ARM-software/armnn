#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/reference)
list(APPEND armnnLibraries armnnRefBackend)

if(ARMNNREF)
    list(APPEND armnnLibraries armnnRefBackendWorkloads)
    list(APPEND armnnUnitTestLibraries armnnRefBackendUnitTests)
else()
    message(STATUS "Reference backend is disabled")
endif()
