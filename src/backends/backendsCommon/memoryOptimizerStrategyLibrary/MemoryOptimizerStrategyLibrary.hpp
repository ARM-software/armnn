//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IMemoryOptimizerStrategy.hpp>
#include "MemoryOptimizerStrategyFactory.hpp"
#include <algorithm>

#include "strategies/ConstantMemoryStrategy.hpp"
#include "strategies/StrategyValidator.hpp"
#include "strategies/SingleAxisPriorityList.hpp"

namespace
{
// Default Memory Optimizer Strategies
static const std::vector<std::string> memoryOptimizationStrategies(
{
    "ConstantMemoryStrategy",
    "SingleAxisPriorityList"
    "StrategyValidator"
});

#define CREATE_MEMORY_OPTIMIZER_STRATEGY(strategyName, memoryOptimizerStrategy)                                  \
{                                                                                                                \
    MemoryOptimizerStrategyFactory memoryOptimizerStrategyFactory;                                               \
    memoryOptimizerStrategy = memoryOptimizerStrategyFactory.CreateMemoryOptimizerStrategy<strategyName>();      \
}                                                                                                                \

} // anonymous namespace
namespace armnn
{
    std::unique_ptr<IMemoryOptimizerStrategy> GetMemoryOptimizerStrategy(const std::string& strategyName)
    {
        auto doesStrategyExist = std::find(memoryOptimizationStrategies.begin(),
                                           memoryOptimizationStrategies.end(),
                                           strategyName) != memoryOptimizationStrategies.end();
        if (doesStrategyExist)
        {
            std::unique_ptr<IMemoryOptimizerStrategy> memoryOptimizerStrategy = nullptr;
            CREATE_MEMORY_OPTIMIZER_STRATEGY(armnn::ConstantMemoryStrategy,
                                             memoryOptimizerStrategy);
            return  memoryOptimizerStrategy;
        }
        return nullptr;
    }


    const std::vector<std::string>& GetMemoryOptimizerStrategyNames()
    {
        return memoryOptimizationStrategies;
    }
} // namespace armnn