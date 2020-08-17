#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnNeonBackendUnitTests_sources
    NeonCreateWorkloadTests.cpp
    NeonEndToEndTests.cpp
    NeonJsonPrinterTests.cpp
    NeonLayerSupportTests.cpp
    NeonLayerTests.cpp
    NeonOptimizedNetworkTests.cpp
    NeonRuntimeTests.cpp
    NeonTensorHandleTests.cpp
    NeonTimerTest.cpp
    NeonWorkloadFactoryHelper.hpp
)

if (ARMNNREF)
    list(APPEND armnnNeonBackendUnitTests_sources
        NeonFallbackTests.cpp
        NeonMemCopyTests.cpp
        )
endif()

add_library(armnnNeonBackendUnitTests OBJECT ${armnnNeonBackendUnitTests_sources})
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)

