//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/memoryOptimizationStrategies/ConstLayerMemoryOptimizerStrategy.hpp>

#include <doctest/doctest.h>
#include <vector>

using namespace armnn;

TEST_SUITE("ConstLayerMemoryOptimizerStrategyTestSuite")
{

TEST_CASE("ConstLayerMemoryOptimizerStrategyTest")
{
    // create a few memory blocks
    MemBlock memBlock0(0, 2, 20, 0, 0);
    MemBlock memBlock1(2, 3, 10, 20, 1);
    MemBlock memBlock2(3, 5, 15, 30, 2);
    MemBlock memBlock3(5, 6, 20, 50, 3);
    MemBlock memBlock4(7, 8, 5, 70, 4);

    std::vector<MemBlock> memBlocks;
    memBlocks.reserve(5);
    memBlocks.push_back(memBlock0);
    memBlocks.push_back(memBlock1);
    memBlocks.push_back(memBlock2);
    memBlocks.push_back(memBlock3);
    memBlocks.push_back(memBlock4);

    // Optimize the memory blocks with ConstLayerMemoryOptimizerStrategy
    ConstLayerMemoryOptimizerStrategy constLayerMemoryOptimizerStrategy;
    CHECK_EQ(constLayerMemoryOptimizerStrategy.GetName(), std::string("ConstLayerMemoryOptimizerStrategy"));
    CHECK_EQ(constLayerMemoryOptimizerStrategy.GetMemBlockStrategyType(), MemBlockStrategyType::SingleAxisPacking);
    auto memBins = constLayerMemoryOptimizerStrategy.Optimize(memBlocks);
    CHECK(memBins.size() == 5);

    CHECK(memBins[1].m_MemBlocks.size() == 1);
    CHECK(memBins[1].m_MemBlocks[0].m_Offset == 0);
    CHECK(memBins[1].m_MemBlocks[0].m_MemSize == 10);
    CHECK(memBins[1].m_MemBlocks[0].m_Index == 1);

    CHECK(memBins[4].m_MemBlocks.size() == 1);
    CHECK(memBins[4].m_MemBlocks[0].m_Offset == 0);
    CHECK(memBins[4].m_MemBlocks[0].m_MemSize == 5);
    CHECK(memBins[4].m_MemBlocks[0].m_Index == 4);
}

}
