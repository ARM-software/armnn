#
# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnMemoryOptimizationStrategiesUnitTests_sources
     ConstLayerMemoryOptimizerStrategyTests.cpp
     MemoryOptimizerStrategyValidatorTests.cpp
)

add_library(armnnMemoryOptimizationStrategiesUnitTests OBJECT ${armnnMemoryOptimizationStrategiesUnitTests_sources})
target_include_directories(armnnMemoryOptimizationStrategiesUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnn)
target_include_directories(armnnMemoryOptimizationStrategiesUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/armnnUtils)
target_include_directories(armnnMemoryOptimizationStrategiesUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
target_include_directories(armnnMemoryOptimizationStrategiesUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/profiling)
target_include_directories(armnnMemoryOptimizationStrategiesUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/profiling/common/include)
target_include_directories(armnnMemoryOptimizationStrategiesUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/third-party)
