#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnNeonBackendUnitTests_sources
    NeonCreateWorkloadTests.cpp
    NeonLayerSupportTests.cpp
    NeonLayerTests.cpp
)

add_library(armnnNeonBackendUnitTests STATIC ${armnnNeonBackendUnitTests_sources})
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnNeonBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)