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

BOOST_AUTO_TEST_SUITE(NeonEndToEnd)

std::vector<armnn::BackendId> defaultBackends = {armnn::Compute::CpuAcc};

BOOST_AUTO_TEST_CASE(ConstantUsage_Neon_Float32)
{
    BOOST_TEST(ConstantUsageFloat32Test(defaultBackends));
}

#if defined(ARMNNREF_ENABLED)

// This test unit needs the reference backend, it's not available if the reference backend is not built

BOOST_AUTO_TEST_CASE(FallbackToCpuRef)
{
    using namespace armnn;

    // Create runtime in which test will run and allow fallback to CpuRef.
    IRuntime::CreationOptions options;
    IRuntimePtr runtime(IRuntime::Create(options));

    // Builds up the structure of the network.
    INetworkPtr net(INetwork::Create());

    IConnectableLayer* input = net->AddInputLayer(0);

    // This layer configuration isn't supported by CpuAcc but we allow fallback to CpuRef so it shoud pass.
    NormalizationDescriptor descriptor;
    IConnectableLayer* pooling = net->AddNormalizationLayer(descriptor);

    IConnectableLayer* output = net->AddOutputLayer(0);

    input->GetOutputSlot(0).Connect(pooling->GetInputSlot(0));
    pooling->GetOutputSlot(0).Connect(output->GetInputSlot(0));

    input->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));
    pooling->GetOutputSlot(0).SetTensorInfo(TensorInfo({ 1, 1, 4, 4 }, DataType::Float32));

    // optimize the network
    std::vector<BackendId> backends = {Compute::CpuAcc, Compute::CpuRef};
    IOptimizedNetworkPtr optNet = Optimize(*net, backends, runtime->GetDeviceSpec());

    // Load it into the runtime. It should pass.
    NetworkId netId;
    BOOST_TEST(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);
}

#endif

BOOST_AUTO_TEST_CASE(NeonGreaterSimpleEndToEndTest)
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ArithmeticSimpleEndToEnd<armnn::DataType::Float32, armnn::DataType::Boolean>(defaultBackends,
                                                                                 LayerType::Greater,
                                                                                 expectedOutput);
}

BOOST_AUTO_TEST_CASE(NeonGreaterSimpleEndToEndUint8Test)
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ArithmeticSimpleEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(defaultBackends,
                                                                                         LayerType::Greater,
                                                                                         expectedOutput);
}

BOOST_AUTO_TEST_CASE(NeonGreaterBroadcastEndToEndTest)
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ArithmeticBroadcastEndToEnd<armnn::DataType::Float32, armnn::DataType::Boolean>(defaultBackends,
                                                                                    LayerType::Greater,
                                                                                    expectedOutput);
}

BOOST_AUTO_TEST_CASE(NeonGreaterBroadcastEndToEndUint8Test)
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ArithmeticBroadcastEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Boolean>(defaultBackends,
                                                                                            LayerType::Greater,
                                                                                            expectedOutput);
}

BOOST_AUTO_TEST_CASE(NeonConcatEndToEndDim0Test)
{
    ConcatDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonConcatEndToEndDim0Uint8Test)
{
    ConcatDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonConcatEndToEndDim1Test)
{
    ConcatDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonConcatEndToEndDim1Uint8Test)
{
    ConcatDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonConcatEndToEndDim3Test)
{
    ConcatDim3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonConcatEndToEndDim3Uint8Test)
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

BOOST_AUTO_TEST_CASE(NeonPreluEndToEndFloat32Test)
{
    PreluEndToEndNegativeTest<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonPreluEndToEndTestUint8Test)
{
    PreluEndToEndPositiveTest<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSpaceToDepthNHWCEndToEndTest1)
{
    SpaceToDepthNHWCEndToEndTest1(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSpaceToDepthNCHWEndToEndTest1)
{
    SpaceToDepthNCHWEndToEndTest1(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSpaceToDepthNHWCEndToEndTest2)
{
    SpaceToDepthNHWCEndToEndTest2(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSpaceToDepthNCHWEndToEndTest2)
{
    SpaceToDepthNCHWEndToEndTest2(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter1dEndToEndTest)
{
    Splitter1dEndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter1dEndToEndUint8Test)
{
    Splitter1dEndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter2dDim0EndToEndTest)
{
    Splitter2dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter2dDim1EndToEndTest)
{
    Splitter2dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter2dDim0EndToEndUint8Test)
{
    Splitter2dDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter2dDim1EndToEndUint8Test)
{
    Splitter2dDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter3dDim0EndToEndTest)
{
    Splitter3dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter3dDim1EndToEndTest)
{
    Splitter3dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter3dDim2EndToEndTest)
{
    Splitter3dDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter3dDim0EndToEndUint8Test)
{
    Splitter3dDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter3dDim1EndToEndUint8Test)
{
    Splitter3dDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter3dDim2EndToEndUint8Test)
{
    Splitter3dDim2EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter4dDim0EndToEndTest)
{
    Splitter4dDim0EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter4dDim1EndToEndTest)
{
    Splitter4dDim1EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter4dDim2EndToEndTest)
{
    Splitter4dDim2EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter4dDim3EndToEndTest)
{
    Splitter4dDim3EndToEnd<armnn::DataType::Float32>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter4dDim0EndToEndUint8Test)
{
    Splitter4dDim0EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter4dDim1EndToEndUint8Test)
{
    Splitter4dDim1EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter4dDim2EndToEndUint8Test)
{
    Splitter4dDim2EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonSplitter4dDim3EndToEndUint8Test)
{
    Splitter4dDim3EndToEnd<armnn::DataType::QuantisedAsymm8>(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonQuantizedLstmEndToEndTest)
{
    QuantizedLstmEndToEnd(defaultBackends);
}

BOOST_AUTO_TEST_CASE(NeonTransposeConvolution2dEndToEndFloatNchwTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(NeonTransposeConvolution2dEndToEndUint8NchwTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NCHW);
}

BOOST_AUTO_TEST_CASE(NeonTransposeConvolution2dEndToEndFloatNhwcTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_CASE(NeonTransposeConvolution2dEndToEndUint8NhwcTest)
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QuantisedAsymm8, armnn::DataType::Signed32>(
        defaultBackends, armnn::DataLayout::NHWC);
}

BOOST_AUTO_TEST_SUITE_END()
