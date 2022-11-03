#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnTosaRefBackendUnitTests_sources
        TosaRefEndToEndTests.cpp
        TosaRefLayerTests.cpp
        TosaRefLayerSupportTests.cpp
        TosaRefOptimizedNetworkTests.cpp
)

add_library(armnnTosaRefBackendUnitTests OBJECT ${armnnTosaRefBackendUnitTests_sources})
target_include_directories(armnnTosaRefBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnTosaRefBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnTosaRefBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnTestUtils)
target_include_directories(armnnTosaRefBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnTosaRefBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/third-party/doctest)
