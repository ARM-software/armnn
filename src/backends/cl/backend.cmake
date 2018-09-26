#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

if(ARMCOMPUTECL)
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/cl)
    list(APPEND armnnLibraries armnnClBackend armnnClBackendWorkloads)
else()
    message("CL backend is disabled")
    add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/cl)
    list(APPEND armnnLibraries armnnClBackend)
endif()
