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

BOOST_AUTO_TEST_CASE(IsLayerSupportedBFloat16Reference)
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::BFloat16>(&factory);
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
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::QAsymmU8>(&factory);
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedInt8Reference)
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::QSymmS8>(&factory);
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedInt16Reference)
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::QSymmS16>(&factory);
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

BOOST_AUTO_TEST_CASE(IsConvertBf16ToFp32SupportedReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertBf16ToFp32Layer,
      armnn::DataType::BFloat16, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_CASE(IsConvertBf16ToFp32SupportedFp32InputReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertBf16ToFp32Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Reference for ConvertBf16ToFp32 layer: input type not supported\n");
}

BOOST_AUTO_TEST_CASE(IsConvertBf16ToFp32SupportedBf16OutputReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertBf16ToFp32Layer,
      armnn::DataType::BFloat16, armnn::DataType::BFloat16>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Reference for ConvertBf16ToFp32 layer: output type not supported\n");
}

BOOST_AUTO_TEST_CASE(IsConvertFp32ToBf16SupportedReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp32ToBf16Layer,
      armnn::DataType::Float32, armnn::DataType::BFloat16>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_CASE(IsConvertFp32ToBf16SupportedBf16InputReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp32ToBf16Layer,
      armnn::DataType::BFloat16, armnn::DataType::BFloat16>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Reference for ConvertFp32ToBf16 layer: input type not supported\n");
}

BOOST_AUTO_TEST_CASE(IsConvertFp32ToBf16SupportedFp32OutputReference)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp32ToBf16Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Reference for ConvertFp32ToBf16 layer: output type not supported\n");
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

    BOOST_CHECK(reasonIfUnsupported.find(
        "Reference Mean: Expected 4 dimensions but got 2 dimensions instead, for the 'output' tensor.")
        != std::string::npos);
}

BOOST_AUTO_TEST_CASE(IsConstantSupportedRef)
{
    std::string reasonIfUnsupported;

    bool result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Float16>(reasonIfUnsupported);
    BOOST_CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Float32>(reasonIfUnsupported);
    BOOST_CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::QAsymmU8>(reasonIfUnsupported);
    BOOST_CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Boolean>(reasonIfUnsupported);
    BOOST_CHECK(!result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::QSymmS16>(reasonIfUnsupported);
    BOOST_CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::QSymmS8>(reasonIfUnsupported);
    BOOST_CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::QAsymmS8>(reasonIfUnsupported);
    BOOST_CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::BFloat16>(reasonIfUnsupported);
    BOOST_CHECK(result);
}

BOOST_AUTO_TEST_SUITE_END()
