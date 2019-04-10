//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>

#include <ExecutionFrame.hpp>

// Test that the values set in m_NextExecutionFrame are correct.
// The execution order is given by the m_NextExecutionFrame in each ExecutionFrame.
// A
// |
// B
// |
// C
BOOST_AUTO_TEST_CASE(NextExecutionFrameTest)
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

    BOOST_CHECK_EQUAL(nextExecutionFrameA, &executionFrameB);
    BOOST_CHECK_EQUAL(nextExecutionFrameB, &executionFrameC);

    BOOST_CHECK(!nextExecutionFrameC);

    BOOST_CHECK_NE(nextExecutionFrameA, &executionFrameC);
}

