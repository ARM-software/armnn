//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "../NeonWorkloadFactory.hpp"
#include <neon/NeonBackend.hpp>
#include <armnnTestUtils/LayerTestResult.hpp>
#include <armnnTestUtils/MemCopyTestImpl.hpp>
#include <armnnTestUtils/MockBackend.hpp>
#include <doctest/doctest.h>

namespace
{

template <>
struct MemCopyTestHelper<armnn::NeonWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::NeonBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::NeonWorkloadFactory GetFactory(
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ModelOptions& modelOptions = {})
    {
        armnn::NeonBackend backend;
        return armnn::NeonWorkloadFactory(armnn::PolymorphicPointerDowncast<armnn::NeonMemoryManager>(memoryManager),
                                          backend.CreateBackendSpecificModelContext(modelOptions));
    }
};
}    // namespace


TEST_SUITE("NeonMemCopy")
{
TEST_CASE("CopyBetweenCpuAndNeon")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::MockWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenNeonAndCpu")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenCpuAndNeonWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::MockWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenNeonAndCpuWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

}
