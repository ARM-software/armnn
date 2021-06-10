//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <aclCommon/test/MemCopyTestImpl.hpp>

#include <neon/NeonWorkloadFactory.hpp>

#include <reference/RefWorkloadFactory.hpp>
#include <reference/test/RefWorkloadFactoryHelper.hpp>

#include <doctest/doctest.h>

TEST_SUITE("NeonMemCopy")
{
TEST_CASE("CopyBetweenCpuAndNeon")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenNeonAndCpu")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenCpuAndNeonWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::RefWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenNeonAndCpuWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::RefWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

}
