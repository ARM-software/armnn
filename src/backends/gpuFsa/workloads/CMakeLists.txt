#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnGpuFsaBackendWorkloads_sources
        GpuFsaBaseWorkload.hpp
)

add_library(armnnGpuFsaBackendWorkloads OBJECT ${armnnGpuFsaBackendWorkloads_sources})
target_include_directories(armnnGpuFsaBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnGpuFsaBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnGpuFsaBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnGpuFsaBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnGpuFsaBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnGpuFsaBackendWorkloads PRIVATE ${PROJECT_SOURCE_DIR}/profiling/client/include)
