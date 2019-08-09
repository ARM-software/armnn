//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/test/EndToEndTestImpl.hpp>

#include <backendsCommon/test/ArithmeticTestImpl.hpp>
#include <backendsCommon/test/ConcatTestImpl.hpp>
#include <backendsCommon/test/DequantizeEndToEndTestImpl.hpp>
#include <backendsCommon/test/PreluEndToEndTestImpl.hpp>
#include <backendsCommon/test/QuantizedLstmEndToEndTestImpl.hpp>
#include <backendsCommon/test/SpaceToDepthEndToEndTestImpl.hpp>
#include <backendsCommon/test/SplitterEndToEndTestImpl.hpp>
#include <backendsCommon/test/TransposeConvolution2dEndToEndTestImpl.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(ClEndToEnd)

std::vector<armnn::BackendId> defaultBackends = {armnn::Compute::GpuAcc};

BOOST_AUTO_TEST_CASE(ConstantUsage_Cl_Float32)
{
    ConstantUsageFloat32Test(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClConcatEndToEndDim0Test)
{
    ConcatDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClConcatEndToEndDim0Uint8Test)
{
    ConcatDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClConcatEndToEndDim1Test)
{
    ConcatDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClConcatEndToEndDim1Uint8Test)
{
    ConcatDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClConcatEndToEndDim3Test)
{
    ConcatDim3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClConcatEndToEndDim3Uint8Test)
{
    ConcatDim3EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(DequantizeEndToEndSimpleTest)
{
    DequantizeEndToEndSimple<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(DequantizeEndToEndOffsetTest)
{
    DequantizeEndToEndOffset<armnn::DataType::QuantisedAsymm8>(defaultBackends);
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

BOOST_AUTO_TEST_CASE(ClPreluEndToEndFloat32Test)
{
    PreluEndToEndNegativeTest<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClPreluEndToEndTestUint8)
{
    PreluEndToEndPositiveTest<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSpaceToDepthNHWCEndToEndTest1)
{
    SpaceToDepthNHWCEndToEndTest1(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSpaceToDepthNCHWEndToEndTest1)
{
    SpaceToDepthNCHWEndToEndTest1(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSpaceToDepthNHWCEndToEndTest2)
{
    SpaceToDepthNHWCEndToEndTest2(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSpaceToDepthNCHWEndToEndTest2)
{
    SpaceToDepthNCHWEndToEndTest2(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter1dEndToEndTest)
{
    Splitter1dEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter1dEndToEndUint8Test)
{
    Splitter1dEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter2dDim0EndToEndTest)
{
    Splitter2dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter2dDim1EndToEndTest)
{
    Splitter2dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter2dDim0EndToEndUint8Test)
{
    Splitter2dDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter2dDim1EndToEndUint8Test)
{
    Splitter2dDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter3dDim0EndToEndTest)
{
    Splitter3dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter3dDim1EndToEndTest)
{
    Splitter3dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter3dDim2EndToEndTest)
{
    Splitter3dDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter3dDim0EndToEndUint8Test)
{
    Splitter3dDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter3dDim1EndToEndUint8Test)
{
    Splitter3dDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter3dDim2EndToEndUint8Test)
{
    Splitter3dDim2EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter4dDim0EndToEndTest)
{
    Splitter4dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter4dDim1EndToEndTest)
{
    Splitter4dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter4dDim2EndToEndTest)
{
    Splitter4dDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter4dDim3EndToEndTest)
{
    Splitter4dDim3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter4dDim0EndToEndUint8Test)
{
    Splitter4dDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter4dDim1EndToEndUint8Test)
{
    Splitter4dDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter4dDim2EndToEndUint8Test)
{
    Splitter4dDim2EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(ClSplitter4dDim3EndToEndUint8Test)
{
    Splitter4dDim3EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

// TransposeConvolution2d
BOOST_AUTO_TEST_CASE(ClTransposeConvolution2dEndToEndFloatNchwTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(ClTransposeConvolution2dEndToEndUint8NchwTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(ClTransposeConvolution2dEndToEndFloatNhwcTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(ClTransposeConvolution2dEndToEndUint8NhwcTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(ClQuantizedLstmEndToEndTest)
{
    QuantizedLstmEndToEnd(defaultBackends);
}

BOOST_AUTO_TEST_SUITE_END()