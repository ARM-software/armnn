//
// Copyright © 2017-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <backendsCommon/test/EndToEndTestImpl.hpp>

#include <backendsCommon/test/ActivationEndToEndTestImpl.hpp>
#include <backendsCommon/test/AdditionEndToEndTestImpl.hpp>
#include <backendsCommon/test/ArgMinMaxEndToEndTestImpl.hpp>
#include <backendsCommon/test/BatchMatMulEndToEndTestImpl.hpp>
#include <backendsCommon/test/ComparisonEndToEndTestImpl.hpp>
#include <backendsCommon/test/ConcatEndToEndTestImpl.hpp>
#include <backendsCommon/test/DepthToSpaceEndToEndTestImpl.hpp>
#include <backendsCommon/test/DequantizeEndToEndTestImpl.hpp>
#include <backendsCommon/test/DetectionPostProcessEndToEndTestImpl.hpp>
#include <backendsCommon/test/ElementwiseBinaryEndToEndTestImpl.hpp>
#include <backendsCommon/test/ElementwiseUnaryEndToEndTestImpl.hpp>
#include <backendsCommon/test/FillEndToEndTestImpl.hpp>
#include <backendsCommon/test/GatherNdEndToEndTestImpl.hpp>
#include <backendsCommon/test/InstanceNormalizationEndToEndTestImpl.hpp>
#include "backendsCommon/test/Pooling2dEndToEndTestImpl.hpp"
#include <backendsCommon/test/PreluEndToEndTestImpl.hpp>
#include <backendsCommon/test/QLstmEndToEndTestImpl.hpp>
#include <backendsCommon/test/QuantizedLstmEndToEndTestImpl.hpp>
#include <backendsCommon/test/ReduceEndToEndTestImpl.hpp>
#include <backendsCommon/test/ReshapeEndToEndTestImpl.hpp>
#include <backendsCommon/test/ResizeEndToEndTestImpl.hpp>
#include <backendsCommon/test/ReverseV2EndToEndTestImpl.hpp>
#include <backendsCommon/test/SliceEndToEndTestImpl.hpp>
#include <backendsCommon/test/SpaceToDepthEndToEndTestImpl.hpp>
#include <backendsCommon/test/SplitterEndToEndTestImpl.hpp>
#include <backendsCommon/test/StridedSliceEndToEndTestImpl.hpp>
#include <backendsCommon/test/SubgraphUtilsTest.hpp>
#include <backendsCommon/test/TileEndToEndTestImpl.hpp>
#include <backendsCommon/test/TransposeConvolution2dEndToEndTestImpl.hpp>
#include <backendsCommon/test/TransposeEndToEndTestImpl.hpp>

#include <doctest/doctest.h>

TEST_SUITE("NeonEndToEnd")
{
std::vector<armnn::BackendId> neonDefaultBackends = {armnn::Compute::CpuAcc};

// ElementwiseUnary
// Abs
TEST_CASE("NeonAbsEndToEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(neonDefaultBackends,
                                                             UnaryOperation::Abs);
}
// Rsqrt
TEST_CASE("NeonRsqrtEndToEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(neonDefaultBackends,
                                                             UnaryOperation::Rsqrt);
}

// Constant
TEST_CASE("ConstantUsage_Neon_Float32")
{
    CHECK(ConstantUsageFloat32Test(neonDefaultBackends));
}

#if defined(ARMNNREF_ENABLED)

// This test unit needs the reference backend, it's not available if the reference backend is not built

TEST_CASE("FallbackToCpuRef")
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
    CHECK(runtime->LoadNetwork(netId, std::move(optNet)) == Status::Success);
}

#endif

TEST_CASE("NeonGreaterSimpleEndToEndTest")
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ComparisonSimpleEndToEnd<armnn::DataType::Float32>(neonDefaultBackends,
                                                       ComparisonOperation::Greater,
                                                       expectedOutput);
}

TEST_CASE("NeonGreaterSimpleEndToEndUint8Test")
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ComparisonSimpleEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends,
                                                               ComparisonOperation::Greater,
                                                               expectedOutput);
}

TEST_CASE("NeonGreaterBroadcastEndToEndTest")
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ComparisonBroadcastEndToEnd<armnn::DataType::Float32>(neonDefaultBackends,
                                                          ComparisonOperation::Greater,
                                                          expectedOutput);
}

TEST_CASE("NeonGreaterBroadcastEndToEndUint8Test")
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ComparisonBroadcastEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends,
                                                                  ComparisonOperation::Greater,
                                                                  expectedOutput);
}

// ElementwiseBinary
// Add
TEST_CASE("NeonAdditionEndToEndFloat32Test")
{
    AdditionEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonAdditionEndToEndUint8Test")
{
    AdditionEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonAdditionEndToEndFloat32Simple3DTest")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, BinaryOperation::Add);
}
TEST_CASE("NeonAdditionEndToEndFloat16Simple3DTest")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float16>(neonDefaultBackends, BinaryOperation::Add);
}

// Div
TEST_CASE("NeonDivEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, BinaryOperation::Div);
}

// Mul
TEST_CASE("NeonMulEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, BinaryOperation::Mul);
}
TEST_CASE("NeonMulEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends, BinaryOperation::Mul);
}

// Sub
TEST_CASE("NeonSubtractionEndToEndFloat32Simple3DTest")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, BinaryOperation::Sub);
}
TEST_CASE("NeonSubtractionEndToEndFloat16Simple3DTest")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float16>(neonDefaultBackends, BinaryOperation::Sub);
}

// Power
TEST_CASE("NeonPowerEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, BinaryOperation::Power);
}

// SqDiff
TEST_CASE("NeonSquaredDifferenceEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, BinaryOperation::SqDiff);
}

TEST_CASE("NeonSquaredDifferenceEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends, BinaryOperation::SqDiff);
}

// Batch Mat Mul
TEST_CASE("NeonBatchMatMulEndToEndFloat32Test")
{
    BatchMatMulEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonBatchMatMulEndToEndInt8Test")
{
    BatchMatMulEndToEnd<armnn::DataType::QAsymmS8>(neonDefaultBackends);
}

// Concat
TEST_CASE("NeonConcatEndToEndDim0Test")
{
    ConcatDim0EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonConcatEndToEndDim0Uint8Test")
{
    ConcatDim0EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonConcatEndToEndDim1Test")
{
    ConcatDim1EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonConcatEndToEndDim1Uint8Test")
{
    ConcatDim1EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonConcatEndToEndDim3Test")
{
    ConcatDim3EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonConcatEndToEndDim3Uint8Test")
{
    ConcatDim3EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

// DepthToSpace
TEST_CASE("NeonDephtToSpaceEndToEndNchwFloat32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("NeonDephtToSpaceEndToEndNchwFloat16")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float16>(neonDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("NeonDephtToSpaceEndToEndNchwUint8")
{
    DepthToSpaceEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("NeonDephtToSpaceEndToEndNchwInt16")
{
    DepthToSpaceEndToEnd<armnn::DataType::QSymmS16>(neonDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("NeonDephtToSpaceEndToEndNchwSigned32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Signed32>(neonDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("NeonDephtToSpaceEndToEndNhwcFloat32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("NeonDephtToSpaceEndToEndNhwcFloat16")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float16>(neonDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("NeonDephtToSpaceEndToEndNhwcUint8")
{
    DepthToSpaceEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("NeonDephtToSpaceEndToEndNhwcInt16")
{
    DepthToSpaceEndToEnd<armnn::DataType::QSymmS16>(neonDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("NeonDephtToSpaceEndToEndNhwcSigned32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Signed32>(neonDefaultBackends, armnn::DataLayout::NHWC);
}

// Dequantize
TEST_CASE("DequantizeEndToEndSimpleTest")
{
    DequantizeEndToEndSimple<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("DequantizeEndToEndOffsetTest")
{
    DequantizeEndToEndOffset<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

// GatherNd
TEST_CASE("NeonGatherNdFloatTest")
{
    GatherNdEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonGatherNdUint8Test")
{
    GatherNdEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonGatherNdInt16Test")
{
    GatherNdEndToEnd<armnn::DataType::QSymmS16>(neonDefaultBackends);
}

TEST_CASE("NeonGatherNdMultiDimFloatTest")
{
    GatherNdMultiDimEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonGatherNdMultiDimUint8Test")
{
    GatherNdMultiDimEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonGatherNdMultiDimInt16Test")
{
    GatherNdMultiDimEndToEnd<armnn::DataType::QSymmS16>(neonDefaultBackends);
}

// Activations
// Linear
TEST_CASE("NeonLinearEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(neonDefaultBackends, ActivationFunction::Linear);
}

// Sigmoid
TEST_CASE("NeonSigmoidEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(neonDefaultBackends, ActivationFunction::Sigmoid);
}

// ReLu
TEST_CASE("NeonReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(neonDefaultBackends, ActivationFunction::ReLu);
}

// BoundedReLu
TEST_CASE("NeonBoundedReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(neonDefaultBackends, ActivationFunction::BoundedReLu);
}

// LeakyRelu
TEST_CASE("NeonLeakyReluActivationFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(neonDefaultBackends, ActivationFunction::LeakyReLu, 1.f, 0,  0.01f);
}

// Elu
TEST_CASE("NeonEluEndToEndTestFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(neonDefaultBackends, ActivationFunction::Elu);
}

// HardSwish
TEST_CASE("NeonHardSwishEndToEndTestFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(neonDefaultBackends, ActivationFunction::HardSwish);
}

// TanH
TEST_CASE("NeonTanHEndToEndTestFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(neonDefaultBackends, ActivationFunction::TanH, 1.f, 0, 2, 3);
}

// Prelu
TEST_CASE("NeonPreluEndToEndFloat32Test")
{
    PreluEndToEndNegativeTest<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonPreluEndToEndTestUint8Test")
{
    PreluEndToEndPositiveTest<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

// SpaceToDepth
TEST_CASE("NeonSpaceToDepthNhwcEndToEndTest1")
{
    SpaceToDepthNhwcEndToEndTest1(neonDefaultBackends);
}

TEST_CASE("NeonSpaceToDepthNchwEndToEndTest1")
{
    SpaceToDepthNchwEndToEndTest1(neonDefaultBackends);
}

TEST_CASE("NeonSpaceToDepthNhwcEndToEndTest2")
{
    SpaceToDepthNhwcEndToEndTest2(neonDefaultBackends);
}

TEST_CASE("NeonSpaceToDepthNchwEndToEndTest2")
{
    SpaceToDepthNchwEndToEndTest2(neonDefaultBackends);
}

// Split
TEST_CASE("NeonSplitter1dEndToEndTest")
{
    Splitter1dEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter1dEndToEndUint8Test")
{
    Splitter1dEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter2dDim0EndToEndTest")
{
    Splitter2dDim0EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter2dDim1EndToEndTest")
{
    Splitter2dDim1EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter2dDim0EndToEndUint8Test")
{
    Splitter2dDim0EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter2dDim1EndToEndUint8Test")
{
    Splitter2dDim1EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter3dDim0EndToEndTest")
{
    Splitter3dDim0EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter3dDim1EndToEndTest")
{
    Splitter3dDim1EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter3dDim2EndToEndTest")
{
    Splitter3dDim2EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter3dDim0EndToEndUint8Test")
{
    Splitter3dDim0EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter3dDim1EndToEndUint8Test")
{
    Splitter3dDim1EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter3dDim2EndToEndUint8Test")
{
    Splitter3dDim2EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter4dDim0EndToEndTest")
{
    Splitter4dDim0EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter4dDim1EndToEndTest")
{
    Splitter4dDim1EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter4dDim2EndToEndTest")
{
    Splitter4dDim2EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter4dDim3EndToEndTest")
{
    Splitter4dDim3EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter4dDim0EndToEndUint8Test")
{
    Splitter4dDim0EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter4dDim1EndToEndUint8Test")
{
    Splitter4dDim1EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter4dDim2EndToEndUint8Test")
{
    Splitter4dDim2EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonSplitter4dDim3EndToEndUint8Test")
{
    Splitter4dDim3EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

// Tile
TEST_CASE("NeonTileEndToEndFloat32")
{
    TileEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}
TEST_CASE("NeonTileEndToEndFloat16")
{
    TileEndToEnd<armnn::DataType::Float16>(neonDefaultBackends);
}
TEST_CASE("NeonTileEndToEndQAsymmS8")
{
    TileEndToEnd<armnn::DataType::QAsymmS8>(neonDefaultBackends);
}
TEST_CASE("NeonTileEndToEndQAsymmU8")
{
    TileEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}
TEST_CASE("NeonTileEndToEndQSymmS8")
{
    TileEndToEnd<armnn::DataType::QSymmS8>(neonDefaultBackends);
}
TEST_CASE("NeonTileEndToEndQSymmS16")
{
    TileEndToEnd<armnn::DataType::QSymmS16>(neonDefaultBackends);
}
TEST_CASE("NeonTileEndToEndSigned32")
{
    TileEndToEnd<armnn::DataType::Signed32>(neonDefaultBackends);
}

TEST_CASE("NeonQuantizedLstmEndToEndTest")
{
    QuantizedLstmEndToEnd(neonDefaultBackends);
}

TEST_CASE("NeonTransposeConvolution2dEndToEndFloatNchwTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        neonDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("NeonTransposeConvolution2dEndToEndUint8NchwTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        neonDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("NeonTransposeConvolution2dEndToEndFloatNhwcTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        neonDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("NeonTransposeConvolution2dEndToEndUint8NhwcTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        neonDefaultBackends, armnn::DataLayout::NHWC);
}

// Transpose
TEST_CASE("NeonTransposeEndToEndTest")
{
    TransposeEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonImportNonAlignedInputPointerTest")
{
    ImportNonAlignedInputPointerTest(neonDefaultBackends);
}

TEST_CASE("NeonExportNonAlignedOutputPointerTest")
{
    ExportNonAlignedOutputPointerTest(neonDefaultBackends);
}

TEST_CASE("NeonImportAlignedPointerTest")
{
    ImportAlignedPointerTest(neonDefaultBackends);
}

TEST_CASE("NeonImportOnlyWorkload")
{
    ImportOnlyWorkload(neonDefaultBackends);
}

TEST_CASE("NeonExportOnlyWorkload")
{
    ExportOnlyWorkload(neonDefaultBackends);
}

TEST_CASE("NeonImportAndExportWorkload")
{
    ImportAndExportWorkload(neonDefaultBackends);
}

TEST_CASE("NeonExportOutputWithSeveralOutputSlotConnectionsTest")
{
    ExportOutputWithSeveralOutputSlotConnectionsTest(neonDefaultBackends);
}

// InstanceNormalization
TEST_CASE("NeonInstanceNormalizationNchwEndToEndTest1")
{
    InstanceNormalizationNchwEndToEndTest1(neonDefaultBackends);
}

TEST_CASE("NeonInstanceNormalizationNchwEndToEndTest2")
{
    InstanceNormalizationNchwEndToEndTest2(neonDefaultBackends);
}

// Pooling 2D
// Average Pool 2D
TEST_CASE("NeonAvgPool2DEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonAvgPool2DEndtoEndTestFloat16")
{
    AvgPool2dEndToEndFloat16<DataType::Float16>(neonDefaultBackends);
}

TEST_CASE("NeonAvgPool2DIgnoreValueEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(neonDefaultBackends, PaddingMethod::IgnoreValue);
}

// Max Pool 2D
TEST_CASE("NeonMaxPool2DEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonMaxPool2DEndtoEndTestFloat16")
{
    MaxPool2dEndToEndFloat16<DataType::Float16>(neonDefaultBackends);
}

TEST_CASE("NeonMaxPool2DIgnoreValueEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(neonDefaultBackends, PaddingMethod::IgnoreValue);
}

TEST_CASE("NeonMaxPool2DTwoLayerEndtoEndTestFloat32")
{
    MaxPool2dTwoLayerEndToEnd<DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonMaxPool2DThreeLayerEndtoEndTestFloat32")
{
    MaxPool2dThreeLayerEndToEnd<DataType::Float32>(neonDefaultBackends);
}

// Fill
TEST_CASE("NeonFillEndToEndTest")
{
    FillEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonFillEndToEndTestFloat16")
{
    FillEndToEnd<armnn::DataType::Float16>(neonDefaultBackends);
}

TEST_CASE("NeonFillEndToEndTestInt32")
{
    FillEndToEnd<armnn::DataType::Signed32>(neonDefaultBackends);
}

// ArgMinMax
TEST_CASE("NeonArgMaxSimpleTest")
{
    ArgMaxEndToEndSimple<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinSimpleTest")
{
    ArgMinEndToEndSimple<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMaxAxis0Test")
{
    ArgMaxAxis0EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinAxis0Test")
{
    ArgMinAxis0EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMaxAxis1Test")
{
    ArgMaxAxis1EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinAxis1Test")
{
    ArgMinAxis1EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMaxAxis2Test")
{
    ArgMaxAxis2EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinAxis2Test")
{
    ArgMinAxis2EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMaxAxis3Test")
{
    ArgMaxAxis3EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinAxis3Test")
{
    ArgMinAxis3EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonArgMaxSimpleTestQuantisedAsymm8")
{
    ArgMaxEndToEndSimple<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinSimpleTestQuantisedAsymm8")
{
    ArgMinEndToEndSimple<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonArgMaxAxis0TestQuantisedAsymm8")
{
    ArgMaxAxis0EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinAxis0TestQuantisedAsymm8")
{
    ArgMinAxis0EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonArgMaxAxis1TestQuantisedAsymm8")
{
    ArgMaxAxis1EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinAxis1TestQuantisedAsymm8")
{
    ArgMinAxis1EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonArgMaxAxis2TestQuantisedAsymm8")
{
    ArgMaxAxis2EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinAxis2TestQuantisedAsymm8")
{
    ArgMinAxis2EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonArgMaxAxis3TestQuantisedAsymm8")
{
    ArgMaxAxis3EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonArgMinAxis3TestQuantisedAsymm8")
{
    ArgMinAxis3EndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

// Reduce
// Reduce Sum
TEST_CASE("NeonReduceSum2dEndtoEndTestSigned32")
{
    ReduceEndToEnd2d<DataType::Signed32>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum2dEndtoEndTestSigned32WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Signed32>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum2dEndtoEndTestFloat16")
{
    ReduceEndToEnd2d<DataType::Float16>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum2dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Float16>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum2dEndtoEndTestFloat32")
{
    ReduceEndToEnd2d<DataType::Float32>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum2dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Float32>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum2dEndtoEndTestInt8")
{
    ReduceEndToEnd2d<DataType::QAsymmS8>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum2dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd2d<DataType::QAsymmS8>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum3dEndtoEndTestSigned32")
{
    ReduceEndToEnd3d<DataType::Signed32>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum3dEndtoEndTestSigned32WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Signed32>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum3dEndtoEndTestFloat16")
{
    ReduceEndToEnd3d<DataType::Float16>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum3dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Float16>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum3dEndtoEndTestFloat32")
{
    ReduceEndToEnd3d<DataType::Float32>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum3dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Float32>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum3dEndtoEndTestInt8")
{
    ReduceEndToEnd3d<DataType::QAsymmS8>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum3dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd3d<DataType::QAsymmS8>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum4dEndtoEndTestSigned32")
{
    ReduceEndToEnd4d<DataType::Signed32>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum4dEndtoEndTestSigned32WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Signed32>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum4dEndtoEndTestFloat16")
{
    ReduceEndToEnd4d<DataType::Float16>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum4dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Float16>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum4dEndtoEndTestFloat32")
{
    ReduceEndToEnd4d<DataType::Float32>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum4dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Float32>(neonDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("NeonReduceSum4dEndtoEndTestInt8")
{
    ReduceEndToEnd4d<DataType::QAsymmS8>(neonDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("NeonReduceSum4dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd4d<DataType::QAsymmS8>(neonDefaultBackends, ReduceOperation::Sum, true);
}

// Reshape
TEST_CASE("NeonReshapeEndToEndTest")
{
    ReshapeEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonReshapeEndToEndTestFloat16")
{
    ReshapeEndToEndFloat16<armnn::DataType::Float16>(neonDefaultBackends);
}

TEST_CASE("NeonReshapeEndToEndTestInt32")
{
    ReshapeEndToEnd<armnn::DataType::Signed32>(neonDefaultBackends);
}

TEST_CASE("NeonReshapeEndToEndTestInt16")
{
    ReshapeEndToEnd<armnn::DataType::QSymmS16>(neonDefaultBackends);
}

TEST_CASE("NeonReshapeEndToEndTestUInt8")
{
    ReshapeEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends);
}

TEST_CASE("NeonReshapeEndToEndTestInt8")
{
    ReshapeEndToEnd<armnn::DataType::QAsymmS8>(neonDefaultBackends);
}

// Resize Bilinear
TEST_CASE("NeonResizeBilinearEndToEndFloatNchwTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("NeonResizeBilinearEndToEndFloatNhwcTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, armnn::DataLayout::NHWC);
}

// Resize NearestNeighbor
TEST_CASE("NeonResizeNearestNeighborEndToEndFloatNchwTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("NeonResizeNearestNeighborEndToEndFloatNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("NeonResizeNearestNeighborEndToEndFloatAlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("NeonResizeNearestNeighborEndToEndFloatHalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

// ReverseV2
TEST_CASE("NeonReverseV2EndToEndTest")
{
    ReverseV2EndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonReverseV2EndToEndSigned32Test")
{
    ReverseV2EndToEnd<armnn::DataType::Signed32>(neonDefaultBackends);
}

// Slice
TEST_CASE("NeonSliceEndtoEndTestFloat32")
{
    SliceEndToEnd<DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonSliceEndtoEndTestInt32")
{
    SliceEndToEnd<DataType::Signed32>(neonDefaultBackends);
}

TEST_CASE("NeonSliceEndtoEndTestFloat16")
{
    SliceEndToEndFloat16<DataType::Float16>(neonDefaultBackends);
}

TEST_CASE("NeonStridedSliceInvalidSliceEndToEndTest")
{
    StridedSliceInvalidSliceEndToEndTest(neonDefaultBackends);
}

TEST_CASE("NeonForceImportWithAlignedBuffersEndToEndTest"
          // Currently, the Neon workload for activation does not support tensor handle replacement so this test case
          // will always fail.
          * doctest::skip(true))
{
    ForceImportWithAlignedBuffersEndToEndTest(neonDefaultBackends);
}

TEST_CASE("NeonForceImportWithMisalignedInputBuffersEndToEndTest"
          // Currently, the Neon workload for activation does not support tensor handle replacement so this test case
          // will always fail.
          * doctest::skip(true))
{
    ForceImportWithMisalignedInputBuffersEndToEndTest(neonDefaultBackends);
}

TEST_CASE("NeonForceImportWithMisalignedOutputBuffersEndToEndTest"
          // Currently, the Neon workload for activation does not support tensor handle replacement so this test case
          // will always fail.
          * doctest::skip(true))
{
    ForceImportWithMisalignedOutputBuffersEndToEndTest(neonDefaultBackends);
}

TEST_CASE("NeonForceImportWithMisalignedInputAndOutputBuffersEndToEndTest")
{
    ForceImportWithMisalignedInputAndOutputBuffersEndToEndTest(neonDefaultBackends);
}

// DISABLED
//TEST_CASE("NeonDetectionPostProcessRegularNmsTest")
//{
//    std::vector<float> boxEncodings({
//                                        0.0f, 0.0f, 0.0f, 0.0f,
//                                        0.0f, 1.0f, 0.0f, 0.0f,
//                                        0.0f, -1.0f, 0.0f, 0.0f,
//                                        0.0f, 0.0f, 0.0f, 0.0f,
//                                        0.0f, 1.0f, 0.0f, 0.0f,
//                                        0.0f, 0.0f, 0.0f, 0.0f
//                                    });
//    std::vector<float> scores({
//                                  0.0f, 0.9f, 0.8f,
//                                  0.0f, 0.75f, 0.72f,
//                                  0.0f, 0.6f, 0.5f,
//                                  0.0f, 0.93f, 0.95f,
//                                  0.0f, 0.5f, 0.4f,
//                                  0.0f, 0.3f, 0.2f
//                              });
//    std::vector<float> anchors({
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 10.5f, 1.0f, 1.0f,
//                                   0.5f, 10.5f, 1.0f, 1.0f,
//                                   0.5f, 100.5f, 1.0f, 1.0f
//                               });
//    DetectionPostProcessRegularNmsEndToEnd<armnn::DataType::Float32>(neonDefaultBackends,
//                                                                     boxEncodings,
//                                                                     scores,
//                                                                     anchors);
//}

inline void QuantizeData(uint8_t* quant, const float* dequant, const TensorInfo& info)
{
    for (size_t i = 0; i < info.GetNumElements(); i++)
    {
        quant[i] = armnn::Quantize<uint8_t>(dequant[i], info.GetQuantizationScale(), info.GetQuantizationOffset());
    }
}

// DISABLED
//TEST_CASE("NeonDetectionPostProcessRegularNmsUint8Test")
//{
//    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, armnn::DataType::Float32);
//    armnn::TensorInfo scoresInfo({ 1, 6, 3 }, armnn::DataType::Float32);
//    armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::Float32);
//
//    boxEncodingsInfo.SetQuantizationScale(1.0f);
//    boxEncodingsInfo.SetQuantizationOffset(1);
//    scoresInfo.SetQuantizationScale(0.01f);
//    scoresInfo.SetQuantizationOffset(0);
//    anchorsInfo.SetQuantizationScale(0.5f);
//    anchorsInfo.SetQuantizationOffset(0);
//
//    std::vector<float> boxEncodings({
//                                        0.0f, 0.0f, 0.0f, 0.0f,
//                                        0.0f, 1.0f, 0.0f, 0.0f,
//                                        0.0f, -1.0f, 0.0f, 0.0f,
//                                        0.0f, 0.0f, 0.0f, 0.0f,
//                                        0.0f, 1.0f, 0.0f, 0.0f,
//                                        0.0f, 0.0f, 0.0f, 0.0f
//                                    });
//    std::vector<float> scores({
//                                  0.0f, 0.9f, 0.8f,
//                                  0.0f, 0.75f, 0.72f,
//                                  0.0f, 0.6f, 0.5f,
//                                  0.0f, 0.93f, 0.95f,
//                                  0.0f, 0.5f, 0.4f,
//                                  0.0f, 0.3f, 0.2f
//                              });
//    std::vector<float> anchors({
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 10.5f, 1.0f, 1.0f,
//                                   0.5f, 10.5f, 1.0f, 1.0f,
//                                   0.5f, 100.5f, 1.0f, 1.0f
//                               });
//
//    std::vector<uint8_t> qBoxEncodings(boxEncodings.size(), 0);
//    std::vector<uint8_t> qScores(scores.size(), 0);
//    std::vector<uint8_t> qAnchors(anchors.size(), 0);
//    QuantizeData(qBoxEncodings.data(), boxEncodings.data(), boxEncodingsInfo);
//    QuantizeData(qScores.data(), scores.data(), scoresInfo);
//    QuantizeData(qAnchors.data(), anchors.data(), anchorsInfo);
//    DetectionPostProcessRegularNmsEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends, qBoxEncodings,
//                                                                             qScores, qAnchors,
//                                                                             1.0f, 1, 0.01f, 0, 0.5f, 0);
//}
//
//TEST_CASE("NeonDetectionPostProcessFastNmsTest")
//{
//    std::vector<float> boxEncodings({
//                                        0.0f, 0.0f, 0.0f, 0.0f,
//                                        0.0f, 1.0f, 0.0f, 0.0f,
//                                        0.0f, -1.0f, 0.0f, 0.0f,
//                                        0.0f, 0.0f, 0.0f, 0.0f,
//                                        0.0f, 1.0f, 0.0f, 0.0f,
//                                        0.0f, 0.0f, 0.0f, 0.0f
//                                    });
//    std::vector<float> scores({
//                                  0.0f, 0.9f, 0.8f,
//                                  0.0f, 0.75f, 0.72f,
//                                  0.0f, 0.6f, 0.5f,
//                                  0.0f, 0.93f, 0.95f,
//                                  0.0f, 0.5f, 0.4f,
//                                  0.0f, 0.3f, 0.2f
//                              });
//    std::vector<float> anchors({
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 10.5f, 1.0f, 1.0f,
//                                   0.5f, 10.5f, 1.0f, 1.0f,
//                                   0.5f, 100.5f, 1.0f, 1.0f
//                               });
//    DetectionPostProcessFastNmsEndToEnd<armnn::DataType::Float32>(neonDefaultBackends,
//                                                                  boxEncodings,
//                                                                  scores,
//                                                                  anchors);
//}
//
// DISABLED
//TEST_CASE("NeonDetectionPostProcessFastNmsUint8Test")
//{
//    armnn::TensorInfo boxEncodingsInfo({ 1, 6, 4 }, armnn::DataType::Float32);
//    armnn::TensorInfo scoresInfo({ 1, 6, 3 }, armnn::DataType::Float32);
//    armnn::TensorInfo anchorsInfo({ 6, 4 }, armnn::DataType::Float32);
//
//    boxEncodingsInfo.SetQuantizationScale(1.0f);
//    boxEncodingsInfo.SetQuantizationOffset(1);
//    scoresInfo.SetQuantizationScale(0.01f);
//    scoresInfo.SetQuantizationOffset(0);
//    anchorsInfo.SetQuantizationScale(0.5f);
//    anchorsInfo.SetQuantizationOffset(0);
//
//    std::vector<float> boxEncodings({
//                                        0.0f, 0.0f, 0.0f, 0.0f,
//                                        0.0f, 1.0f, 0.0f, 0.0f,
//                                        0.0f, -1.0f, 0.0f, 0.0f,
//                                        0.0f, 0.0f, 0.0f, 0.0f,
//                                        0.0f, 1.0f, 0.0f, 0.0f,
//                                        0.0f, 0.0f, 0.0f, 0.0f
//                                    });
//    std::vector<float> scores({
//                                  0.0f, 0.9f, 0.8f,
//                                  0.0f, 0.75f, 0.72f,
//                                  0.0f, 0.6f, 0.5f,
//                                  0.0f, 0.93f, 0.95f,
//                                  0.0f, 0.5f, 0.4f,
//                                  0.0f, 0.3f, 0.2f
//                              });
//    std::vector<float> anchors({
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 0.5f, 1.0f, 1.0f,
//                                   0.5f, 10.5f, 1.0f, 1.0f,
//                                   0.5f, 10.5f, 1.0f, 1.0f,
//                                   0.5f, 100.5f, 1.0f, 1.0f
//                               });
//
//    std::vector<uint8_t> qBoxEncodings(boxEncodings.size(), 0);
//    std::vector<uint8_t> qScores(scores.size(), 0);
//    std::vector<uint8_t> qAnchors(anchors.size(), 0);
//    QuantizeData(qBoxEncodings.data(), boxEncodings.data(), boxEncodingsInfo);
//    QuantizeData(qScores.data(), scores.data(), scoresInfo);
//    QuantizeData(qAnchors.data(), anchors.data(), anchorsInfo);
//    DetectionPostProcessFastNmsEndToEnd<armnn::DataType::QAsymmU8>(neonDefaultBackends, qBoxEncodings,
//                                                                          qScores, qAnchors,
//                                                                          1.0f, 1, 0.01f, 0, 0.5f, 0);
//}

TEST_CASE("NeonQLstmEndToEndTest")
{
    QLstmEndToEnd(neonDefaultBackends);
}

TEST_CASE("NeonReshapeRemovalSimpleCaseEndToEnd")
{
    ReshapeRemovalEndToEnd<armnn::DataType::Float32>(neonDefaultBackends);
}

TEST_CASE("NeonReshapeRemovalNCHWFirstEndToEnd")
{
    ReshapeRemovalNCHWEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, false, true);
}

TEST_CASE("NeonReshapeRemovalNCHWSecondEndToEnd")
{
    ReshapeRemovalNCHWEndToEnd<armnn::DataType::Float32>(neonDefaultBackends, false, false);
}

}
