//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnnTestUtils/LayerTestResult.hpp>
#include <armnnTestUtils/MemCopyTestImpl.hpp>
#include <armnnTestUtils/MockBackend.hpp>
#include <cl/ClWorkloadFactory.hpp>
#include <cl/ClBackend.hpp>
#include <doctest/doctest.h>

namespace
{

template <>
struct MemCopyTestHelper<armnn::ClWorkloadFactory>
{
    static armnn::IBackendInternal::IMemoryManagerSharedPtr GetMemoryManager()
    {
        armnn::ClBackend backend;
        return backend.CreateMemoryManager();
    }

    static armnn::ClWorkloadFactory GetFactory(
        const armnn::IBackendInternal::IMemoryManagerSharedPtr& memoryManager,
        const armnn::ModelOptions& modelOptions = {})
    {
        armnn::ClBackend backend;
        return armnn::ClWorkloadFactory(armnn::PolymorphicPointerDowncast<armnn::ClMemoryManager>(memoryManager),
                                        backend.CreateBackendSpecificModelContext(modelOptions));
    }
};
}    // namespace

TEST_SUITE("ClMemCopy")
{
TEST_CASE("CopyBetweenCpuAndGpu")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::MockWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenGpuAndCpu")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenCpuAndGpuWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::MockWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE("CopyBetweenGpuAndCpuWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::MockWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData,  result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

}
