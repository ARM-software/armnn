//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <layers/ConvertFp16ToFp32Layer.hpp>
#include <layers/ConvertFp32ToFp16Layer.hpp>
#include <armnnTestUtils/TensorHelpers.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <neon/NeonWorkloadFactory.hpp>
#include <backendsCommon/test/IsLayerSupportedTestImpl.hpp>
#include <backendsCommon/test/LayerTests.hpp>

#include <doctest/doctest.h>

#include <string>

TEST_SUITE("NeonLayerSupport")
{
TEST_CASE("IsLayerSupportedFloat16Neon")
{
    armnn::NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::Float16>(&factory);
}

TEST_CASE("IsLayerSupportedFloat32Neon")
{
    armnn::NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::Float32>(&factory);
}

TEST_CASE("IsLayerSupportedQAsymmU8Neon")
{
    armnn::NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::QAsymmU8>(&factory);
}

TEST_CASE("IsLayerSupportedQAsymmS8Neon")
{
    armnn::NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::QAsymmS8>(&factory);
}

TEST_CASE("IsLayerSupportedQSymmS8Neon")
{
    armnn::NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::QSymmS8>(&factory);
}

TEST_CASE("IsConvertFp16ToFp32SupportedNeon")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsConvertFp32ToFp16SupportedNeon")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsLogicalBinarySupportedNeon")
{
    std::string reasonIfUnsupported;

    bool result = IsLogicalBinaryLayerSupportedTests<armnn::NeonWorkloadFactory,
      armnn::DataType::Boolean, armnn::DataType::Boolean>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsLogicalBinaryBroadcastSupportedNeon")
{
    std::string reasonIfUnsupported;

    bool result = IsLogicalBinaryLayerBroadcastSupportedTests<armnn::NeonWorkloadFactory,
      armnn::DataType::Boolean, armnn::DataType::Boolean>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsMeanSupportedNeon")
{
    std::string reasonIfUnsupported;

    bool result = IsMeanLayerSupportedTests<armnn::NeonWorkloadFactory,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsConstantSupportedNeon")
{
    std::string reasonIfUnsupported;

    bool result = IsConstantLayerSupportedTests<armnn::NeonWorkloadFactory,
            armnn::DataType::Float16>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::NeonWorkloadFactory,
            armnn::DataType::Float32>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::NeonWorkloadFactory,
            armnn::DataType::QAsymmU8>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::NeonWorkloadFactory,
            armnn::DataType::Boolean>(reasonIfUnsupported);
    CHECK(!result);

    result = IsConstantLayerSupportedTests<armnn::NeonWorkloadFactory,
            armnn::DataType::QSymmS16>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::NeonWorkloadFactory,
            armnn::DataType::QSymmS8>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::NeonWorkloadFactory,
            armnn::DataType::QAsymmS8>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::NeonWorkloadFactory,
            armnn::DataType::BFloat16>(reasonIfUnsupported);
    CHECK(result);
}

}
