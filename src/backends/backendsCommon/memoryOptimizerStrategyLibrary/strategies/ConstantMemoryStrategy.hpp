//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include <armnn/Types.hpp>
#include <armnn/backends/IMemoryOptimizerStrategy.hpp>

namespace armnn
{
// ConstLayerMemoryOptimizer: Create a unique MemBin for each MemBlock and assign it an offset of 0
class ConstantMemoryStrategy : public IMemoryOptimizerStrategy
{
public:
    ConstantMemoryStrategy()
    : m_Name(std::string("ConstantMemoryStrategy"))
    , m_MemBlockStrategyType(MemBlockStrategyType::SingleAxisPacking) {}

    std::string GetName() const override;

    MemBlockStrategyType GetMemBlockStrategyType() const override;

    std::vector<MemBin> Optimize(std::vector<MemBlock>& memBlocks) override;

private:
    std::string m_Name;
    MemBlockStrategyType m_MemBlockStrategyType;
};

} // namespace armnn