#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnAclCommon_sources
    ArmComputeTensorUtils.hpp
    ArmComputeTensorUtils.cpp
    ArmComputeUtils.hpp
)

add_library(armnnAclCommon STATIC ${armnnAclCommon_sources})
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
