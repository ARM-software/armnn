//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <doctest/doctest.h>

#include <ExecutionFrame.hpp>

// Test that the values set in m_NextExecutionFrame are correct.
// The execution order is given by the m_NextExecutionFrame in each ExecutionFrame.
// A
// |
// B
// |
// C
TEST_SUITE("NextExecutionFrameTestSuite")
{
TEST_CASE("NextExecutionFrameTest")
{
    armnn::ExecutionFrame executionFrameA;
    armnn::ExecutionFrame executionFrameB;
    armnn::ExecutionFrame executionFrameC;

    executionFrameA.SetNextExecutionFrame(&executionFrameB);
    executionFrameB.SetNextExecutionFrame(&executionFrameC);
    //not setting C to check that the default setting is nullptr.

    auto nextExecutionFrameA = executionFrameA.ExecuteWorkloads(nullptr);
    auto nextExecutionFrameB = executionFrameB.ExecuteWorkloads(&executionFrameA);
    auto nextExecutionFrameC = executionFrameC.ExecuteWorkloads(&executionFrameB);

    CHECK_EQ(nextExecutionFrameA, &executionFrameB);
    CHECK_EQ(nextExecutionFrameB, &executionFrameC);

    CHECK(!nextExecutionFrameC);

    CHECK_NE(nextExecutionFrameA, &executionFrameC);
}
}

