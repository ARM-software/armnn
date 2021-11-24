#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnAclCommonUnitTests_sources
    ArmComputeTensorUtilsTests.cpp
    CreateWorkloadClNeon.hpp
    MemCopyTests.cpp
    MemCopyTestImpl.hpp
)

add_library(armnnAclCommonUnitTests OBJECT ${armnnAclCommonUnitTests_sources})
target_include_directories(armnnAclCommonUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnAclCommonUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnAclCommonUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnTestUtils)
target_include_directories(armnnAclCommonUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnAclCommonUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnAclCommonUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
