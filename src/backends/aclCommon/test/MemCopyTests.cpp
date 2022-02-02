//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <aclCommon/ArmComputeTensorUtils.hpp>

#if defined(ARMCOMPUTECL_ENABLED) && defined(ARMCOMPUTENEON_ENABLED)
#include <armnnTestUtils/LayerTestResult.hpp>
#include <armnnTestUtils/MemCopyTestImpl.hpp>
#include <cl/ClWorkloadFactory.hpp>
#include <cl/test/ClContextControlFixture.hpp>
#include <cl/test/ClWorkloadFactoryHelper.hpp>
#include <neon/NeonWorkloadFactory.hpp>
#include <neon/test/NeonWorkloadFactoryHelper.hpp>
#endif

#include <doctest/doctest.h>

TEST_SUITE("MemCopyCommon")
{
TEST_CASE("AclTypeConversions")
{
    arm_compute::Strides strides(1, 2, 3, 4);
    armnn::TensorShape convertedStrides = armnn::armcomputetensorutils::GetStrides(strides);

    CHECK(convertedStrides[0] == 4);
    CHECK(convertedStrides[1] == 3);
    CHECK(convertedStrides[2] == 2);
    CHECK(convertedStrides[3] == 1);

    arm_compute::TensorShape shape(5, 6, 7, 8);
    armnn::TensorShape convertedshape = armnn::armcomputetensorutils::GetShape(shape);

    CHECK(convertedshape[0] == 8);
    CHECK(convertedshape[1] == 7);
    CHECK(convertedshape[2] == 6);
    CHECK(convertedshape[3] == 5);
}

}

#if defined(ARMCOMPUTECL_ENABLED) && defined(ARMCOMPUTENEON_ENABLED)

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



TEST_CASE_FIXTURE(ClContextControlFixture, "CopyBetweenNeonAndGpu")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData, result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CopyBetweenGpuAndNeon")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(false);
    auto predResult = CompareTensors(result.m_ActualData, result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CopyBetweenNeonAndGpuWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::NeonWorkloadFactory, armnn::ClWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData, result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

TEST_CASE_FIXTURE(ClContextControlFixture, "CopyBetweenGpuAndNeonWithSubtensors")
{
    LayerTestResult<float, 4> result =
        MemCopyTest<armnn::ClWorkloadFactory, armnn::NeonWorkloadFactory, armnn::DataType::Float32>(true);
    auto predResult = CompareTensors(result.m_ActualData, result.m_ExpectedData,
                                     result.m_ActualShape, result.m_ExpectedShape);
    CHECK_MESSAGE(predResult.m_Result, predResult.m_Message.str());
}

#endif
