//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/test/EndToEndTestImpl.hpp>
#include <backendsCommon/test/MergerTestImpl.hpp>
#include <backendsCommon/test/ArithmeticTestImpl.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ClEndToEnd)

std::vector<armnn::BackendId> defaultBackends = {armnn::Compute::GpuAcc};

BOOST_AUTO_TEST_CASE(ConstantUsage_Cl_Float32)
{
    ConstantUsageFloat32Test(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClMergerEndToEndDim0Test)
{
    MergerDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClMergerEndToEndDim0Uint8Test)
{
    MergerDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClMergerEndToEndDim1Test)
{
    MergerDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClMergerEndToEndDim1Uint8Test)
{
    MergerDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClMergerEndToEndDim3Test)
{
    MergerDim3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClMergerEndToEndDim3Uint8Test)
{
    MergerDim3EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClGreaterSimpleEndToEndTest)
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ArithmeticSimpleEndToEnd<armnn::DataType::Float32, armnn::DataType::Boolean>(defaultBackends,
                                                                                 LayerType::Greater,
                                                                                 expectedOutput);
}

BOOST_AUTO_TEST_CASE(ClGreaterSimpleEndToEndUint8Test)
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ArithmeticSimpleEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(defaultBackends,
                                                                                         LayerType::Greater,
                                                                                         expectedOutput);
}

BOOST_AUTO_TEST_CASE(ClGreaterBroadcastEndToEndTest)
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ArithmeticBroadcastEndToEnd<armnn::DataType::Float32, armnn::DataType::Boolean>(defaultBackends,
                                                                                    LayerType::Greater,
                                                                                    expectedOutput);
}

BOOST_AUTO_TEST_CASE(ClGreaterBroadcastEndToEndUint8Test)
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ArithmeticBroadcastEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(defaultBackends,
                                                                                            LayerType::Greater,
                                                                                            expectedOutput);
}


BOOST_AUTO_TEST_SUITE_END()
