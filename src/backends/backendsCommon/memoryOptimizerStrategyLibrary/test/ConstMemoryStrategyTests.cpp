//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/memoryOptimizerStrategyLibrary/strategies/ConstantMemoryStrategy.hpp>
#include <backendsCommon/memoryOptimizerStrategyLibrary/strategies/StrategyValidator.hpp>

#include <doctest/doctest.h>
#include <vector>

using namespace armnn;

TEST_SUITE("ConstMemoryStrategyTestSuite")
{

TEST_CASE("ConstMemoryStrategyTest")
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

    // Optimize the memory blocks with ConstantMemoryStrategy
    ConstantMemoryStrategy constLayerMemoryOptimizerStrategy;
    CHECK_EQ(constLayerMemoryOptimizerStrategy.GetName(), std::string("ConstantMemoryStrategy"));
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

TEST_CASE("ConstLayerMemoryOptimizerStrategyValidatorTest")
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
    auto ptr = std::make_shared<ConstantMemoryStrategy>();
    StrategyValidator validator;
    validator.SetStrategy(ptr);
    // Ensure ConstLayerMemoryOptimizerStrategy is valid
    CHECK_NOTHROW(validator.Optimize(memBlocks));
}

}
