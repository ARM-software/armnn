//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <reference/RefWorkloadFactory.hpp>
#include <reference/RefBackend.hpp>
#include <armnnTestUtils/LayerTestResult.hpp>
#include <armnnTestUtils/MemCopyTestImpl.hpp>
#include <armnnTestUtils/MockBackend.hpp>
#include <doctest/doctest.h>

namespace
{

template <>
struct MemCopyTestHelper<armnn::RefWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::RefBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::RefWorkloadFactory GetFactory(const armnn::IBackendInternal::IMemoryManagerSharedPtr&)
    {
        return armnn::RefWorkloadFactory();
    }
};
}    // namespace

TEST_SUITE("RefMemCopy")
{

    TEST_CASE("CopyBetweenMockAccAndRef")
    {
        LayerTestResult<float, 4> result =
            MemCopyTest<armnn::MockWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(false);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenRefAndMockAcc")
    {
        LayerTestResult<float, 4> result =
            MemCopyTest<armnn::RefWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::Float32>(false);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenMockAccAndRefWithSubtensors")
    {
        LayerTestResult<float, 4> result =
            MemCopyTest<armnn::MockWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(true);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }

    TEST_CASE("CopyBetweenRefAndMockAccWithSubtensors")
    {
        LayerTestResult<float, 4> result =
            MemCopyTest<armnn::RefWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::Float32>(true);
        auto predResult =
            CompareTensors(result.m_ActualData, result.m_ExpectedData, result.m_ActualShape, result.m_ExpectedShape);
        CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
    }
}
