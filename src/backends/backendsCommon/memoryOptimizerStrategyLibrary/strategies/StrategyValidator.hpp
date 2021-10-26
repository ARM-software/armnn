//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/Exceptions.hpp>
#include <armnn/backends/IMemoryOptimizerStrategy.hpp>

namespace armnn
{

class StrategyValidator : public IMemoryOptimizerStrategy
{
public:

    void SetStrategy(std::shared_ptr<IMemoryOptimizerStrategy> strategy)
    {
        m_Strategy = strategy;
        m_MemBlockStrategyType = strategy->GetMemBlockStrategyType();
    }

    std::string GetName() const override
    {
        return "StrategyValidator";
    }

    MemBlockStrategyType GetMemBlockStrategyType() const override
    {
        return m_MemBlockStrategyType;
    }

    std::vector<MemBin> Optimize(std::vector<MemBlock>& memBlocks) override;

private:
    std::shared_ptr<IMemoryOptimizerStrategy> m_Strategy;
    MemBlockStrategyType m_MemBlockStrategyType;
};

} // namespace armnn