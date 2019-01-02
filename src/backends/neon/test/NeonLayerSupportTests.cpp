//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <layers/ConvertFp16ToFp32Layer.hpp>
#include <layers/ConvertFp32ToFp16Layer.hpp>
#include <test/TensorHelpers.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <neon/NeonWorkloadFactory.hpp>
#include <backendsCommon/test/IsLayerSupportedTestImpl.hpp>
#include <backendsCommon/test/LayerTests.hpp>

#include <boost/test/unit_test.hpp>

#include <string>

BOOST_AUTO_TEST_SUITE(NeonLayerSupport)

BOOST_AUTO_TEST_CASE(IsLayerSupportedFloat16Neon)
{
    armnn::NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::Float16>(&factory);
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedFloat32Neon)
{
    armnn::NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::Float32>(&factory);
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedUint8Neon)
{
    armnn::NeonWorkloadFactory factory =
        NeonWorkloadFactoryHelper::GetFactory(NeonWorkloadFactoryHelper::GetMemoryManager());
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::QuantisedAsymm8>(&factory);
}

BOOST_AUTO_TEST_CASE(IsConvertFp16ToFp32SupportedNeon)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_CASE(IsConvertFp32ToFp16SupportedNeon)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float16>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_CASE(IsMeanSupportedNeon)
{
    std::string reasonIfUnsupported;

    bool result = IsMeanLayerSupportedTests<armnn::NeonWorkloadFactory,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_SUITE_END()
