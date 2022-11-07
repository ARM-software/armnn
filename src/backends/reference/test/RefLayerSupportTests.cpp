//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <layers/ConvertFp16ToFp32Layer.hpp>
#include <layers/ConvertFp32ToFp16Layer.hpp>

#include <armnnTestUtils/TensorHelpers.hpp>

#include <armnn/backends/TensorHandle.hpp>
#include <reference/RefWorkloadFactory.hpp>
#include <reference/RefLayerSupport.hpp>
#include <backendsCommon/test/LayerTests.hpp>
#include <backendsCommon/test/IsLayerSupportedTestImpl.hpp>

#include <doctest/doctest.h>

#include <string>

namespace
{

bool LayerTypeMatchesTest()
{
    return LayerTypeMatchesTestImpl<armnn::LayerType::FirstLayer>(Tag<armnn::LayerType::FirstLayer>());
};

} // anonymous namespace

TEST_SUITE("RefLayerSupported")
{
TEST_CASE("IsLayerSupportedLayerTypeMatches")
{
    LayerTypeMatchesTest();
}

TEST_CASE("IsLayerSupportedReferenceAddition")
{
    armnn::TensorShape shape0 = {1,1,3,4};
    armnn::TensorShape shape1 = {4};
    armnn::TensorShape outShape = {1,1,3,4};
    armnn::TensorInfo in0(shape0, armnn::DataType::Float32);
    armnn::TensorInfo in1(shape1, armnn::DataType::Float32);
    armnn::TensorInfo out(outShape, armnn::DataType::Float32);

    armnn::RefLayerSupport supportChecker;
    std::string reasonNotSupported;
    CHECK(supportChecker.IsAdditionSupported(in0, in1, out, reasonNotSupported));
}

TEST_CASE("IsLayerSupportedFloat16Reference")
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::Float16>(&factory);
}

TEST_CASE("IsLayerSupportedFloat32Reference")
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::Float32>(&factory);
}

TEST_CASE("IsLayerSupportedUint8Reference")
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::QAsymmU8>(&factory);
}

TEST_CASE("IsLayerSupportedInt8Reference")
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::QSymmS8>(&factory);
}

TEST_CASE("IsLayerSupportedInt16Reference")
{
    armnn::RefWorkloadFactory factory;
    IsLayerSupportedTests<armnn::RefWorkloadFactory, armnn::DataType::QSymmS16>(&factory);
}

TEST_CASE("IsConvertFp16ToFp32SupportedReference")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsConvertFp16ToFp32SupportedFp32InputReference")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Layer is not supported with float32 data type input");
}

TEST_CASE("IsConvertFp16ToFp32SupportedFp16OutputReference")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Layer is not supported with float16 data type output");
}

TEST_CASE("IsConvertFp32ToFp16SupportedReference")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsConvertFp32ToFp16SupportedFp16InputReference")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float16, armnn::DataType::Float16>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Layer is not supported with float16 data type input");
}

TEST_CASE("IsConvertFp32ToFp16SupportedFp32OutputReference")
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::RefWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(!result);
    CHECK_EQ(reasonIfUnsupported, "Layer is not supported with float32 data type output");
}

TEST_CASE("IsLayerSupportedMeanDimensionsReference")
{
    std::string reasonIfUnsupported;

    bool result = IsMeanLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(result);
}

TEST_CASE("IsLayerNotSupportedMeanDimensionsReference")
{
    std::string reasonIfUnsupported;

    bool result = IsMeanLayerNotSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    CHECK(!result);

    CHECK(reasonIfUnsupported.find(
        "Reference Mean: Expected 4 dimensions but got 2 dimensions instead, for the 'output' tensor.")
        != std::string::npos);
}

TEST_CASE("IsConstantSupportedRef")
{
    std::string reasonIfUnsupported;

    bool result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Float16>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Float32>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::QAsymmU8>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::Boolean>(reasonIfUnsupported);
    CHECK(!result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::QSymmS16>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::QSymmS8>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::QAsymmS8>(reasonIfUnsupported);
    CHECK(result);

    result = IsConstantLayerSupportedTests<armnn::RefWorkloadFactory,
            armnn::DataType::BFloat16>(reasonIfUnsupported);
    CHECK(!result);
    CHECK(reasonIfUnsupported.find("Reference constant: output is not a supported type.") != std::string::npos);

}

}
