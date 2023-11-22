#
# Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnGpuFsaBackendUnitTests_sources
        GpuFsaDefaultAllocatorTests.cpp
        GpuFsaEndToEndTests.cpp
        GpuFsaLayerTests.cpp
        GpuFsaLayerSupportTests.cpp
        GpuFsaOptimizedNetworkTests.cpp
)

add_library(armnnGpuFsaBackendUnitTests OBJECT ${armnnGpuFsaBackendUnitTests_sources})
target_include_directories(armnnGpuFsaBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnGpuFsaBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnGpuFsaBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnTestUtils)
target_include_directories(armnnGpuFsaBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnGpuFsaBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/third-party/doctest)
