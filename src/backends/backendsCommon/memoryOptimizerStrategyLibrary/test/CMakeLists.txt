#
# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnMemoryOptimizationStrategiesUnitTests_sources
            ConstMemoryStrategyTests.cpp
            ValidatorStrategyTests.cpp
            SingleAxisPriorityListTests.cpp
            MemoryOptimizerStrategyLibraryTests.cpp
)

add_library(armnnMemoryOptimizationStrategiesUnitTests OBJECT ${armnnMemoryOptimizationStrategiesUnitTests_sources})
target_include_directories(armnnMemoryOptimizationStrategiesUnitTests PRIVATE ${PROJECT_SOURCE_DIR}/src/backends)
