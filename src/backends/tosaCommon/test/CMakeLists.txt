#
# Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnTosaBackendUnitTests_sources
        OneToManyMappingTests.cpp
        OneToOneMappingTests.cpp
    )

add_library(armnnTosaBackendUnitTests OBJECT ${armnnTosaBackendUnitTests_sources})
target_include_directories(armnnTosaBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnTosaBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnTosaBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnTestUtils)
target_include_directories(armnnTosaBackendUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
