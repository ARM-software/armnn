#
# Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
# SPDX-License-Identifier: MIT
#

add_executable(MemoryStrategyBenchmark
               MemoryStrategyBenchmark.cpp
               TestBlocks.hpp
               TestStrategy.hpp
               TestStrategy.cpp
               ../../include/armnn/Exceptions.hpp
               ../../src/armnn/Exceptions.cpp)

target_link_libraries(MemoryStrategyBenchmark armnnMemoryOptimizationStrategies)
target_include_directories(MemoryStrategyBenchmark PRIVATE
                           ../../include/armnn/backends
                           ../../src/backends/backendsCommon/memoryOptimizerStrategyLibrary
                           ../../third-party/cxxopts)


set_target_properties(MemoryStrategyBenchmark PROPERTIES LINKER_LANGUAGE CXX)
