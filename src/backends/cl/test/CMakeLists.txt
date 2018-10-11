#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnClBackendUnitTests_sources
    ClContextControlFixture.hpp
    ClCreateWorkloadTests.cpp
    ClLayerSupportTests.cpp
    ClLayerTests.cpp
)

add_library(armnnClBackendUnitTests OBJECT ${armnnClBackendUnitTests_sources})
target_include_directories(armnnClBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnClBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnClBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)