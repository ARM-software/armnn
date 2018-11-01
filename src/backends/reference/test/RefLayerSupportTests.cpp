//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <layers/ConvertFp16ToFp32Layer.hpp>
#include <layers/ConvertFp32ToFp16Layer.hpp>
#include <test/TensorHelpers.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <reference/RefWorkloadFactory.hpp>
#include <backendsCommon/test/LayerTests.hpp>
#include <backendsCommon/test/IsLayerSupportedTestImpl.hpp>

#include <boost/test/unit_test.hpp>

#include <string>

namespace
{

bool LayerTypeMatchesTest()
{
    return LayerTypeMatchesTestImpl<armnn::LayerType::FirstLayer>(Tag<armnn::LayerType::FirstLayer>());
};

} // anonymous namespace

BOOST_AUTO_TEST_SUITE(RefLayerSupported)

BOOST_AUTO_TEST_CASE(IsLayerSupportedLayerTypeMatches)
{
    LayerTypeMatchesTest();
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedFloat16Reference)
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::Float16>(&factory);
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedFloat32Reference)
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::Float32>(&factory);
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedUint8Reference)
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::QuantisedAsymm8>(&factory);
}

BOOST_AUTO_TEST_CASE(IsConvertFp16ToFp32SupportedReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_CASE(IsConvertFp16ToFp32SupportedFp32InputReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Layer is not supported with float32 data type input");
}

BOOST_AUTO_TEST_CASE(IsConvertFp16ToFp32SupportedFp16OutputReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float16>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Layer is not supported with float16 data type output");
}

BOOST_AUTO_TEST_CASE(IsConvertFp32ToFp16SupportedReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float16>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_CASE(IsConvertFp32ToFp16SupportedFp16InputReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float16, armnn::DataType::Float16>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Layer is not supported with float16 data type input");
}

BOOST_AUTO_TEST_CASE(IsConvertFp32ToFp16SupportedFp32OutputReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Layer is not supported with float32 data type output");
}

BOOST_AUTO_TEST_SUITE_END()
