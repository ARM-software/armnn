//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>

#include "test/TensorHelpers.hpp"
#include "LayerTests.hpp"

#include "backends/CpuTensorHandle.hpp"
#include "backends/RefWorkloadFactory.hpp"

#include <string>
#include <iostream>
#include <backends/ClWorkloadFactory.hpp>
#include <backends/NeonWorkloadFactory.hpp>

#include "IsLayerSupportedTestImpl.hpp"
#include "ClContextControlFixture.hpp"

#include "layers/ConvertFp16ToFp32Layer.hpp"
#include "layers/ConvertFp32ToFp16Layer.hpp"

BOOST_AUTO_TEST_SUITE(IsLayerSupported)

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

#ifdef ARMCOMPUTENEON_ENABLED
BOOST_AUTO_TEST_CASE(IsLayerSupportedFloat16Neon)
{
    armnn::NeonWorkloadFactory factory;
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::Float16>(&factory);
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedFloat32Neon)
{
    armnn::NeonWorkloadFactory factory;
    IsLayerSupportedTests<armnn::NeonWorkloadFactory, armnn::DataType::Float32>(&factory);
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedUint8Neon)
{
    armnn::NeonWorkloadFactory factory;
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
#endif //#ifdef ARMCOMPUTENEON_ENABLED.


#ifdef ARMCOMPUTECL_ENABLED

BOOST_FIXTURE_TEST_CASE(IsLayerSupportedFloat16Cl, ClContextControlFixture)
{
    armnn::ClWorkloadFactory factory;
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::Float16>(&factory);
}

BOOST_FIXTURE_TEST_CASE(IsLayerSupportedFloat32Cl, ClContextControlFixture)
{
    armnn::ClWorkloadFactory factory;
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::Float32>(&factory);
}

BOOST_FIXTURE_TEST_CASE(IsLayerSupportedUint8Cl, ClContextControlFixture)
{
    armnn::ClWorkloadFactory factory;
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::QuantisedAsymm8>(&factory);
}

BOOST_FIXTURE_TEST_CASE(IsConvertFp16ToFp32SupportedCl, ClContextControlFixture)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_FIXTURE_TEST_CASE(IsConvertFp16ToFp32SupportedFp32InputCl, ClContextControlFixture)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Input should be Float16");
}

BOOST_FIXTURE_TEST_CASE(IsConvertFp16ToFp32SupportedFp16OutputCl, ClContextControlFixture)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp16ToFp32Layer,
      armnn::DataType::Float16, armnn::DataType::Float16>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Output should be Float32");
}

BOOST_FIXTURE_TEST_CASE(IsConvertFp32ToFp16SupportedCl, ClContextControlFixture)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float16>(reasonIfUnsupported);

    BOOST_CHECK(result);
}

BOOST_FIXTURE_TEST_CASE(IsConvertFp32ToFp16SupportedFp16InputCl, ClContextControlFixture)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float16, armnn::DataType::Float16>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Input should be Float32");
}

BOOST_FIXTURE_TEST_CASE(IsConvertFp32ToFp16SupportedFp32OutputCl, ClContextControlFixture)
{
    std::string reasonIfUnsupported;

    bool result = IsConvertLayerSupportedTests<armnn::ClWorkloadFactory, armnn::ConvertFp32ToFp16Layer,
      armnn::DataType::Float32, armnn::DataType::Float32>(reasonIfUnsupported);

    BOOST_CHECK(!result);
    BOOST_CHECK_EQUAL(reasonIfUnsupported, "Output should be Float16");
}
#endif //#ifdef ARMCOMPUTECL_ENABLED.

BOOST_AUTO_TEST_SUITE_END()
