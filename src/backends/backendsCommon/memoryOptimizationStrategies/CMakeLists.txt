#
# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnMemoryOptimizationStrategies_sources
     ConstLayerMemoryOptimizerStrategy.hpp
     ConstLayerMemoryOptimizerStrategy.cpp
     MemoryOptimizerStrategyValidator.hpp
     MemoryOptimizerStrategyValidator.cpp
)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()

add_library(armnnMemoryOptimizationStrategies OBJECT ${armnnMemoryOptimizationStrategies_sources})
target_include_directories(armnnMemoryOptimizationStrategies PRIVATE ${PROJECT_SOURCE_DIR}/include/armnn)
target_include_directories(armnnMemoryOptimizationStrategies PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnMemoryOptimizationStrategies PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnMemoryOptimizationStrategies PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnMemoryOptimizationStrategies PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnMemoryOptimizationStrategies PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)