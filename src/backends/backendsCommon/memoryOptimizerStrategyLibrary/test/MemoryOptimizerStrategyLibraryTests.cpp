//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/memoryOptimizerStrategyLibrary/MemoryOptimizerStrategyLibrary.hpp>

#include <doctest/doctest.h>

using namespace armnn;

TEST_SUITE("StrategyLibraryTestSuite")
{

TEST_CASE("StrategyLibraryTest")
{
    std::vector<std::string> strategyNames = GetMemoryOptimizerStrategyNames();
    CHECK(strategyNames.size() != 0);
    for (const auto& strategyName: strategyNames)
    {
        auto strategy = GetMemoryOptimizerStrategy(strategyName);
        CHECK(strategy);
        CHECK(strategy->GetName() == strategyName);
    }
}

}

