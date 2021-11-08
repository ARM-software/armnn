//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/backends/IMemoryOptimizerStrategy.hpp>
#include "MemoryOptimizerStrategyFactory.hpp"

#include "strategies/ConstantMemoryStrategy.hpp"
#include "strategies/StrategyValidator.hpp"
#include "strategies/SingleAxisPriorityList.hpp"

#include <map>

namespace armnn
{
namespace
{

static std::map<std::string, std::unique_ptr<IMemoryOptimizerStrategyFactory>>& GetStrategyFactories()
{
    static std::map<std::string, std::unique_ptr<IMemoryOptimizerStrategyFactory>> strategies;

    if (strategies.size() == 0)
    {
        strategies["ConstantMemoryStrategy"] = std::make_unique<StrategyFactory<ConstantMemoryStrategy>>();
        strategies["SingleAxisPriorityList"] = std::make_unique<StrategyFactory<SingleAxisPriorityList>>();
        strategies["StrategyValidator"]      = std::make_unique<StrategyFactory<StrategyValidator>>();
    }
    return strategies;
}

} // anonymous namespace

std::unique_ptr<IMemoryOptimizerStrategy> GetMemoryOptimizerStrategy(const std::string& strategyName)
{
     const auto& strategyFactoryMap = GetStrategyFactories();
     auto strategyFactory = strategyFactoryMap.find(strategyName);
     if (strategyFactory != GetStrategyFactories().end())
     {
         return  strategyFactory->second->CreateMemoryOptimizerStrategy();
     }
    return nullptr;
}

const std::vector<std::string> GetMemoryOptimizerStrategyNames()
{
    const auto& strategyFactoryMap = GetStrategyFactories();
    std::vector<std::string> strategyNames;
    for (const auto& strategyFactory : strategyFactoryMap)
    {
        strategyNames.emplace_back(strategyFactory.first);
    }
    return strategyNames;
}
} // namespace armnn