//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>

#include "test/TensorHelpers.hpp"
#include "LayerTests.hpp"

#include "backends/CpuTensorHandle.hpp"
#include "backends/RefWorkloadFactory.hpp"
#include <Layers.hpp>

#include <string>
#include <iostream>
#include <backends/ClWorkloadFactory.hpp>
#include <backends/NeonWorkloadFactory.hpp>

#include "IsLayerSupportedTestImpl.hpp"


BOOST_AUTO_TEST_SUITE(IsLayerSupported)

BOOST_AUTO_TEST_CASE(IsLayerSupportedLayerTypeMatches)
{
    LayerTypeMatchesTest();
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

#ifdef ARMCOMPUTENEON_ENABLED
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
#endif //#ifdef ARMCOMPUTENEON_ENABLED


#ifdef ARMCOMPUTECL_ENABLED
BOOST_AUTO_TEST_CASE(IsLayerSupportedFloat32Cl)
{
    armnn::ClWorkloadFactory factory;
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::Float32>(&factory);
}

BOOST_AUTO_TEST_CASE(IsLayerSupportedUint8Cl)
{
    armnn::ClWorkloadFactory factory;
    IsLayerSupportedTests<armnn::ClWorkloadFactory, armnn::DataType::QuantisedAsymm8>(&factory);
}
#endif //#ifdef ARMCOMPUTECL_ENABLED

BOOST_AUTO_TEST_SUITE_END()