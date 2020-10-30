#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnClBackendUnitTests_sources
    ClContextControlFixture.hpp
    ClCreateWorkloadTests.cpp
    ClEndToEndTests.cpp
    ClJsonPrinterTests.cpp
    ClLayerSupportTests.cpp
    ClLayerTests.cpp
    ClOptimizedNetworkTests.cpp
    ClRuntimeTests.cpp
    ClWorkloadFactoryHelper.hpp
    Fp16SupportTest.cpp
    OpenClTimerTest.cpp
)

if (ARMNNREF)
    list(APPEND armnnClBackendUnitTests_sources
        ClMemCopyTests.cpp
        )
endif()

if (ARMCOMPUTENEON)
    list(APPEND armnnClBackendUnitTests_sources
        ClFallbackTests.cpp
        )
endif()

add_library(armnnClBackendUnitTests OBJECT ${armnnClBackendUnitTests_sources})
target_include_directories(armnnClBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnClBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnClBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnClBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnClBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)

