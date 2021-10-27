#
# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

list(APPEND armnnMemoryOptimizationStrategies_sources
            MemoryOptimizerStrategyLibrary.hpp
            MemoryOptimizerStrategyFactory.hpp
            strategies/ConstantMemoryStrategy.hpp
            strategies/ConstantMemoryStrategy.cpp
            strategies/StrategyValidator.hpp
            strategies/StrategyValidator.cpp
            strategies/SingleAxisPriorityList.hpp
            strategies/SingleAxisPriorityList.cpp
)

if(BUILD_UNIT_TESTS)
    add_subdirectory(test)
endif()

add_library(armnnMemoryOptimizationStrategies OBJECT ${armnnMemoryOptimizationStrategies_sources})