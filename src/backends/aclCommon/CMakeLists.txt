#
# Copyright © 2017, 2023 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnAclCommon_sources
    ArmComputeSubgraphUtils.hpp
    ArmComputeTensorHandle.hpp
    ArmComputeTensorUtils.hpp
    ArmComputeTensorUtils.cpp
    ArmComputeUtils.hpp
    BaseMemoryManager.cpp
    BaseMemoryManager.hpp
    IClTensorHandle.hpp
)

if(ARMCOMPUTECL)
    list(APPEND armnnAclCommon_sources
        ArmComputeTuningUtils.hpp
        ArmComputeTuningUtils.cpp
    )
endif()

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()

add_library(armnnAclCommon OBJECT ${armnnAclCommon_sources})
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/profiling/client/include)
