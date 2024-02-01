#
# Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_subdirectory(${PROJECT_SOURCE_DIR}/src/backends/gpuFsa)
list(APPEND armnnLibraries armnnGpuFsaBackend)

if(ARMCOMPUTEGPUFSA)
    list(APPEND armnnLibraries armnnGpuFsaBackendLayers)
    list(APPEND armnnLibraries armnnGpuFsaBackendWorkloads)
    list(APPEND armnnUnitTestLibraries armnnGpuFsaBackendUnitTests)
else()
    message(STATUS "GPU Dynamic Fusion backend is disabled")
endif()
