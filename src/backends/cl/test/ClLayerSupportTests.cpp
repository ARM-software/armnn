//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClWorkloadFactoryHelper.hpp"

#include <layers/ConvertFp16ToFp32Layer.hpp>
#include <layers/ConvertFp32ToFp16Layer.hpp>
#include <layers/MeanLayer.hpp>
#include <armnnTestUtils/TensorHelpers.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <cl/ClWorkloadFactory.hpp>
#include <cl/test/ClContextControlFixture.hpp>
#include <backendsCommon/test/IsLayerSupportedTestImpl.hpp>
#include <backendsCommon/test/LayerTests.hpp>

#include <doctest/doctest.h>

#include <string>

TEST_SUITE("ClLayerSupport")
{
TEST_CASE_FIXTURE(ClContextControlFixture, "IsLayerSupportedFloat16Cl")
{
    armnn::ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::Float16>(&factory);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsLayerSupportedFloat32Cl")
{
    armnn::ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::Float32>(&factory);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsLayerSupportedQAsymmU8Cl")
{
    armnn::ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::QAsymmU8>(&factory);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsLayerSupportedQAsymmS8Cl")
{
    armnn::ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::QAsymmS8>(&factory);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsLayerSupportedQSymmS8Cl")
{
    armnn::ClWorkloadFactory factory =
        ClWorkloadFactoryHelper::GetFactory(ClWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::QSymmS8>(&factory);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsConvertFp16ToFp32SupportedCl")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsConvertFp16ToFp32SupportedFp32InputCl")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Input should be Float16");
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsConvertFp16ToFp32SupportedFp16OutputCl")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Output should be Float32");
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsConvertFp32ToFp16SupportedCl")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsConvertFp32ToFp16SupportedFp16InputCl")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float16, armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Input should be Float32");
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsConvertFp32ToFp16SupportedFp32OutputCl")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Output should be Float16");
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsLogicalBinarySupportedCl")
{
    std::string reasonIfUnsupported;

    bool result = IsLogicalBinaryLayerSupportedTests<armnn::ClWorkloadFactory,
      armnn::DataType::Boolean, armnn::DataType::Boolean>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsLogicalBinaryBroadcastSupportedCl")
{
    std::string reasonIfUnsupported;

    bool result = IsLogicalBinaryLayerBroadcastSupportedTests<armnn::ClWorkloadFactory,
      armnn::DataType::Boolean, armnn::DataType::Boolean>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE_FIXTURE(ClContextControlFixture, "IsMeanSupportedCl")
{
    std::string reasonIfUnsupported;

    bool result = IsMeanLayerSupportedTests<armnn::ClWorkloadFactory,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsConstantSupportedCl")
{
    std::string reasonIfUnsupported;

    bool result = IsConstantLayerSupportedTests<armnn::ClWorkloadFactory,
            armnn::DataType::Float16>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::ClWorkloadFactory,
            armnn::DataType::Float32>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::ClWorkloadFactory,
            armnn::DataType::QAsymmU8>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::ClWorkloadFactory,
            armnn::DataType::Boolean>(reasonIfUnsupported);
    CHECK(!result);

    result = IsConstantLayerSupportedTests<armnn::ClWorkloadFactory,
            armnn::DataType::QSymmS16>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::ClWorkloadFactory,
            armnn::DataType::QSymmS8>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::ClWorkloadFactory,
            armnn::DataType::QAsymmS8>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::ClWorkloadFactory,
            armnn::DataType::BFloat16>(reasonIfUnsupported);
    CHECK(!result);
}

}
