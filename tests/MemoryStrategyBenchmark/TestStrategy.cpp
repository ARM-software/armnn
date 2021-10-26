//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "TestStrategy.hpp"

namespace armnn
{

    std::string TestStrategy::GetName() const
    {
        return m_Name;
    }

    MemBlockStrategyType TestStrategy::GetMemBlockStrategyType() const
    {
        return m_MemBlockStrategyType;
    }

    // A IMemoryOptimizerStrategy must ensure that
    // 1: All MemBlocks have been assigned to a MemBin
    // 2: No MemBlock is assigned to multiple MemBins
    // 3: No two Memblocks in a MemBin overlap in both the X and Y axis
    std::vector<MemBin> TestStrategy::Optimize(std::vector<MemBlock>& memBlocks)
    {
        std::vector<MemBin> memBins;
        memBins.reserve(memBlocks.size());

        for (auto& memBlock : memBlocks)
        {
            MemBin memBin;
            memBin.m_MemSize = memBlock.m_MemSize;
            memBin.m_MemBlocks.reserve(1);
            memBlock.m_Offset = 0;
            memBin.m_MemBlocks.push_back(memBlock);
            memBins.push_back(memBin);
        }

        return memBins;
    }

} // namespace armnn