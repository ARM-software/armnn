//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <layers/ConvertFp16ToFp32Layer.hpp>
#include <layers/ConvertFp32ToFp16Layer.hpp>
#include <test/TensorHelpers.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <reference/RefWorkloadFactory.hpp>
#include <reference/RefLayerSupport.hpp>
#include <backendsCommon/test/LayerTests.hpp>
#include <backendsCommon/test/IsLayerSupportedTestImpl.hpp>

#include <boost/test/unit_test.hpp>
#include <boost/algorithm/string/trim.hpp>

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
BOOST_AUTO_TEST_CASE(IsLayerSupportedReferenceAddition)
{
    armnn::TensorShape shape0 = {1,1,3,4};
    armnn::TensorShape shape1 = {4};
    armnn::TensorShape outShape = {1,1,3,4};
    armnn::TensorInfo in0(shape0, armnn::DataType::Float32);
    armnn::TensorInfo in1(shape1, armnn::DataType::Float32);
    armnn::TensorInfo out(outShape, armnn::DataType::Float32);

    armnn::RefLayerSupport supportChecker;
    std::string reasonNotSupported;
    BOOST_CHECK(supportChecker.IsAdditionSupported(in0, in1, out, reasonNotSupported));
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

BOOST_AUTO_TEST_CASE(IsLayerSupportedInt16Reference)
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::QuantisedSymm16>(&factory);
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

BOOST_AUTO_TEST_CASE(IsLayerSupportedMeanDimensionsReference)
{
    std::string reasonIfUnsupported;

    bool result = IsMeanLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_CASE(IsLayerNotSupportedMeanDimensionsReference)
{
    std::string reasonIfUnsupported;

    bool result = IsMeanLayerNotSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(!result);

    boost::algorithm::trim(reasonIfUnsupported);
    BOOST_CHECK_EQUAL(reasonIfUnsupported,
                      "Reference Mean: Expected 4 dimensions but got 2 dimensions instead, for the 'output' tensor.");
}

BOOST_AUTO_TEST_SUITE_END()
