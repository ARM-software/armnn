//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "MemoryOptimizerStrategyLibrary.hpp"
#include "MemoryOptimizerStrategyFactory.hpp"

#include <armnn/BackendHelper.hpp>
#include <armnn/Logging.hpp>
#include <armnn/TypesUtils.hpp>

#include <armnn/backends/IMemoryOptimizerStrategy.hpp>

#include <backendsCommon/memoryOptimizationStrategies/ConstLayerMemoryOptimizerStrategy.hpp>

#include <algorithm>

namespace
{
// Default Memory Optimizer Strategies
static const std::vector<std::string> memoryOptimizationStrategies({
    "ConstLayerMemoryOptimizerStrategy",
    });

#define CREATE_MEMORY_OPTIMIZER_STRATEGY(strategyName, memoryOptimizerStrategy)                                  \
{                                                                                                                \
    MemoryOptimizerStrategyFactory memoryOptimizerStrategyFactory;                                               \
    memoryOptimizerStrategy = memoryOptimizerStrategyFactory.CreateMemoryOptimizerStrategy<strategyName>();      \
}                                                                                                                \

} // anonymous namespace

namespace armnn
{

bool MemoryOptimizerStrategyLibrary::SetMemoryOptimizerStrategy(const BackendId& id, const std::string& strategyName)
{
    auto isStrategyExist = std::find(memoryOptimizationStrategies.begin(),
                                     memoryOptimizationStrategies.end(),
                                     strategyName) != memoryOptimizationStrategies.end();
    if (isStrategyExist)
    {
        std::shared_ptr<IMemoryOptimizerStrategy> memoryOptimizerStrategy = nullptr;
        CREATE_MEMORY_OPTIMIZER_STRATEGY(armnn::ConstLayerMemoryOptimizerStrategy,
                                         memoryOptimizerStrategy);
        if (memoryOptimizerStrategy)
        {
            using BackendCapability = BackendOptions::BackendOption;
            auto strategyType = GetMemBlockStrategyTypeName(memoryOptimizerStrategy->GetMemBlockStrategyType());
            BackendCapability memOptimizeStrategyCapability {strategyType, true};
            if (HasCapability(memOptimizeStrategyCapability, id))
            {
                BackendRegistryInstance().RegisterMemoryOptimizerStrategy(id, memoryOptimizerStrategy);
                return true;
            }
            // reset shared_ptr memoryOptimizerStrategy
            memoryOptimizerStrategy.reset();
        }
    }
    ARMNN_LOG(warning) << "Backend "
                       << id
                       << " is not registered as does not support memory optimizer strategy "
                       << strategyName << "  \n";
    return false;
}

} // namespace armnn