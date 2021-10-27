//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/memoryOptimizerStrategyLibrary/strategies/SingleAxisPriorityList.hpp>
#include <backendsCommon/memoryOptimizerStrategyLibrary/strategies/StrategyValidator.hpp>
#include "TestMemBlocks.hpp"

#include <doctest/doctest.h>
#include <vector>

using namespace armnn;

TEST_SUITE("SingleAxisPriorityListTestSuite")
{
    TEST_CASE("SingleAxisPriorityListTest")
    {
        std::vector<MemBlock> memBlocks = fsrcnn;

        auto singleAxisPriorityList = std::make_shared<SingleAxisPriorityList>();

        CHECK_EQ(singleAxisPriorityList->GetName(), std::string("SingleAxisPriorityList"));
        CHECK_EQ(singleAxisPriorityList->GetMemBlockStrategyType(), MemBlockStrategyType::SingleAxisPacking);

        StrategyValidator validator;
        validator.SetStrategy(singleAxisPriorityList);

        std::vector<MemBin> memBins;

        CHECK_NOTHROW(memBins = validator.Optimize(memBlocks));

        size_t minMemSize = GetMinPossibleMemorySize(memBlocks);
        size_t actualSize = 0;
        for (auto memBin : memBins)
        {
            actualSize += memBin.m_MemSize;
        }

        CHECK(minMemSize == actualSize);
    }
}
