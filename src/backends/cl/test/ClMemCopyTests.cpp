//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClWorkloadFactoryHelper.hpp"

#include <cl/ClWorkloadFactory.hpp>
#include <aclCommon/test/MemCopyTestImpl.hpp>

#include <reference/RefWorkloadFactory.hpp>
#include <reference/test/RefWorkloadFactoryHelper.hpp>

#include <doctest/doctest.h>

TEST_SUITE("ClMemCopy")
{
TEST_CASE("CopyBetweenCpuAndGpu")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenGpuAndCpu")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenCpuAndGpuWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenGpuAndCpuWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

}
