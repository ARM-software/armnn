#
# Copyright © 2017 Arm Ltd. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnAclCommon_sources
    ArmComputeTensorUtils.hpp
    ArmComputeTensorUtils.cpp
    ArmComputeUtils.hpp
    memory/BaseMemoryManager.cpp
    memory/BaseMemoryManager.hpp
    memory/BlobLifetimeManager.cpp
    memory/BlobLifetimeManager.hpp
    memory/BlobMemoryPool.cpp
    memory/BlobMemoryPool.hpp
    memory/IMemoryPool.hpp
    memory/IPoolManager.hpp
    memory/OffsetLifetimeManager.cpp
    memory/OffsetLifetimeManager.hpp
    memory/OffsetMemoryPool.cpp
    memory/OffsetMemoryPool.hpp
    memory/PoolManager.cpp
    memory/PoolManager.hpp
)

add_subdirectory(test)

add_library(armnnAclCommon OBJECT ${armnnAclCommon_sources})
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnAclCommon PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
