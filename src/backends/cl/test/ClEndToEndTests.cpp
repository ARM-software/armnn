//
// Copyright © 2017-2024 Arm Ltd. All rights reserved.
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
#include <backendsCommon/test/ElementwiseBinaryEndToEndTestImpl.hpp>
#include <backendsCommon/test/ElementwiseUnaryEndToEndTestImpl.hpp>
#include <backendsCommon/test/FillEndToEndTestImpl.hpp>
#include <backendsCommon/test/InstanceNormalizationEndToEndTestImpl.hpp>
#include "backendsCommon/test/Pooling2dEndToEndTestImpl.hpp"
#include <backendsCommon/test/PreluEndToEndTestImpl.hpp>
#include <backendsCommon/test/QLstmEndToEndTestImpl.hpp>
#include <backendsCommon/test/QuantizedLstmEndToEndTestImpl.hpp>
#include <backendsCommon/test/ReduceEndToEndTestImpl.hpp>
#include <backendsCommon/test/ReshapeEndToEndTestImpl.hpp>
#include <backendsCommon/test/ResizeEndToEndTestImpl.hpp>
#include <backendsCommon/test/ReverseV2EndToEndTestImpl.hpp>
#include <backendsCommon/test/ScatterNdEndToEndTestImpl.hpp>
#include <backendsCommon/test/SliceEndToEndTestImpl.hpp>
#include <backendsCommon/test/SpaceToDepthEndToEndTestImpl.hpp>
#include <backendsCommon/test/SplitterEndToEndTestImpl.hpp>
#include <backendsCommon/test/SubgraphUtilsTest.hpp>
#include <backendsCommon/test/TileEndToEndTestImpl.hpp>
#include <backendsCommon/test/TransposeConvolution2dEndToEndTestImpl.hpp>
#include <backendsCommon/test/TransposeEndToEndTestImpl.hpp>

#include <doctest/doctest.h>

TEST_SUITE("ClEndToEnd")
{
std::vector<armnn::BackendId> clDefaultBackends = {armnn::Compute::GpuAcc};

// Activations
// Linear
TEST_CASE("ClLinearEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(clDefaultBackends, ActivationFunction::Linear);
}

TEST_CASE("ClLinearEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(clDefaultBackends, ActivationFunction::Linear);
}

// Sigmoid
TEST_CASE("ClSigmoidEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(clDefaultBackends, ActivationFunction::Sigmoid);
}

// ReLu
TEST_CASE("ClReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(clDefaultBackends, ActivationFunction::ReLu);
}

TEST_CASE("ClReLuEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(clDefaultBackends, ActivationFunction::ReLu);
}

// BoundedReLu
TEST_CASE("ClBoundedReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(clDefaultBackends, ActivationFunction::BoundedReLu);
}

TEST_CASE("ClBoundedReLuEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(clDefaultBackends, ActivationFunction::BoundedReLu);
}

// SoftReLu
TEST_CASE("ClSoftReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(clDefaultBackends, ActivationFunction::SoftReLu);
}

// LeakyRelu
TEST_CASE("ClLeakyReluActivationFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(clDefaultBackends, ActivationFunction::LeakyReLu, 1.f, 0, 0.01f);
}

TEST_CASE("ClLeakyReluActivationFloat16")
{
    ActivationEndToEndTest<DataType::Float16>(clDefaultBackends, ActivationFunction::LeakyReLu, 0.3f, 5, 0.01f);
}

// Elu
TEST_CASE("ClEluEndToEndTestFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(clDefaultBackends, ActivationFunction::Elu);
}

TEST_CASE("ClEluEndToEndTestFloat16")
{
    ActivationEndToEndTest<DataType::Float16>(clDefaultBackends, ActivationFunction::Elu);
}

// HardSwish
TEST_CASE("ClHardSwishEndToEndTestFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(clDefaultBackends, ActivationFunction::HardSwish);
}

TEST_CASE("ClHardSwishEndToEndTestFloat16")
{
    ActivationEndToEndTest<DataType::Float16>(clDefaultBackends, ActivationFunction::HardSwish);
}

// TanH
TEST_CASE("ClTanHEndToEndTestFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(clDefaultBackends, ActivationFunction::TanH, 1.f, 0, 2, 3);
}

TEST_CASE("ClTanHEndToEndTestFloat16")
{
    ActivationEndToEndTest<DataType::Float16>(clDefaultBackends, ActivationFunction::TanH, 1.f, 0, 2, 3);
}

// ElementwiseUnary
// Abs
TEST_CASE("ClAbsEndToEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(clDefaultBackends, UnaryOperation::Abs);
}
// Rsqrt
TEST_CASE("ClRsqrtEndToEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(clDefaultBackends, UnaryOperation::Rsqrt);
}

// ElementwiseBinary
// Addition
TEST_CASE("ClAdditionEndToEndFloat32Test")
{
    AdditionEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}
TEST_CASE("ClAdditionEndToEndUint8Test")
{
    AdditionEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClAdditionEndToEndFloat32Simple3DTest")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float32>(clDefaultBackends, BinaryOperation::Add);
}
TEST_CASE("ClAdditionEndToEndFloat16Simple3DTest")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float16>(clDefaultBackends, BinaryOperation::Add);
}

// Div
TEST_CASE("ClDivEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(clDefaultBackends, BinaryOperation::Div);
}

// Mul
TEST_CASE("ClMulEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(clDefaultBackends, BinaryOperation::Mul);
}
TEST_CASE("ClMulEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends, BinaryOperation::Mul);
}

// Sub
TEST_CASE("ClSubtractionEndToEndFloat32Simple3DTest")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float32>(clDefaultBackends, BinaryOperation::Sub);
}
TEST_CASE("ClSubtractionEndToEndFloat16Simple3DTest")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float16>(clDefaultBackends, BinaryOperation::Sub);
}

// Max
TEST_CASE("ClMaximumEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(clDefaultBackends, BinaryOperation::Maximum);
}
TEST_CASE("ClMaximumEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends, BinaryOperation::Maximum);
}

// Min
TEST_CASE("ClMinimumEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(clDefaultBackends, BinaryOperation::Minimum);
}
TEST_CASE("ClMinimumEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends, BinaryOperation::Minimum);
}

// Power
TEST_CASE("ClPowerEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(clDefaultBackends, BinaryOperation::Power);
}

// SqDiff
TEST_CASE("ClSquaredDifferenceEndToEndTestFloat32")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::Float32>(clDefaultBackends, BinaryOperation::SqDiff);
}
TEST_CASE("ClSquaredDifferenceEndToEndTestUint8")
{
    ElementwiseBinarySimpleEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends, BinaryOperation::SqDiff);
}

// Batch Mat Mul
TEST_CASE("ClBatchMatMulEndToEndFloat32Test")
{
    BatchMatMulEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClBatchMatMulEndToEndInt8Test")
{
    BatchMatMulEndToEnd<armnn::DataType::QAsymmS8>(clDefaultBackends);
}

// Constant
TEST_CASE("ConstantUsage_Cl_Float32")
{
    ConstantUsageFloat32Test(clDefaultBackends);
}

TEST_CASE("ClConcatEndToEndDim0Test")
{
    ConcatDim0EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClConcatEndToEndDim0Uint8Test")
{
    ConcatDim0EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClConcatEndToEndDim1Test")
{
    ConcatDim1EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClConcatEndToEndDim1Uint8Test")
{
    ConcatDim1EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClConcatEndToEndDim3Test")
{
    ConcatDim3EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClConcatEndToEndDim3Uint8Test")
{
    ConcatDim3EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

// DepthToSpace
TEST_CASE("DephtToSpaceEndToEndNchwFloat32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float32>(clDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("DephtToSpaceEndToEndNchwFloat16")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float16>(clDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("DephtToSpaceEndToEndNchwUint8")
{
    DepthToSpaceEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("DephtToSpaceEndToEndNchwInt16")
{
    DepthToSpaceEndToEnd<armnn::DataType::QSymmS16>(clDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("DephtToSpaceEndToEndNchwSigned32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Signed32>(clDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("DephtToSpaceEndToEndNhwcFloat32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float32>(clDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("DephtToSpaceEndToEndNhwcFloat16")
{
    DepthToSpaceEndToEnd<armnn::DataType::Float16>(clDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("DephtToSpaceEndToEndNhwcUint8")
{
    DepthToSpaceEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("DephtToSpaceEndToEndNhwcInt16")
{
    DepthToSpaceEndToEnd<armnn::DataType::QSymmS16>(clDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("DephtToSpaceEndToEndNhwcSigned32")
{
    DepthToSpaceEndToEnd<armnn::DataType::Signed32>(clDefaultBackends, armnn::DataLayout::NHWC);
}

// Dequantize
TEST_CASE("DequantizeEndToEndSimpleTest")
{
    DequantizeEndToEndSimple<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("DequantizeEndToEndOffsetTest")
{
    DequantizeEndToEndOffset<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

// ScatterNd
TEST_CASE("ClScatterNd1DInputEndToEndFloat32Test")
{
    ScatterNd1DimUpdateWithInputEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClScatterNd1DNoInputEndToEndFloat32Test")
{
    ScatterNd1DimUpdateNoInputEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClScatterNd2DInputEndToEndFloat32Test")
{
    ScatterNd2DimUpdateWithInputEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClScatterNd2DNoInputEndToEndFloat32Test")
{
    ScatterNd2DimUpdateNoInputEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

// Slice
TEST_CASE("ClSliceEndtoEndTestFloat32")
{
    SliceEndToEnd<DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSliceEndtoEndTestInt32")
{
    SliceEndToEnd<DataType::Signed32>(clDefaultBackends);
}

TEST_CASE("ClSliceEndtoEndTestFloat16")
{
    SliceEndToEndFloat16<DataType::Float16>(clDefaultBackends);
}

TEST_CASE("ClStridedSliceInvalidSliceEndToEndTest")
{
    StridedSliceInvalidSliceEndToEndTest(clDefaultBackends);
}

TEST_CASE("ClGreaterSimpleEndToEndTest")
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ComparisonSimpleEndToEnd<armnn::DataType::Float32>(clDefaultBackends,
                                                       ComparisonOperation::Greater,
                                                       expectedOutput);
}

TEST_CASE("ClGreaterSimpleEndToEndUint8Test")
{
    const std::vector<uint8_t> expectedOutput({ 0, 0, 0, 0,  1, 1, 1, 1,
                                                0, 0, 0, 0,  0, 0, 0, 0 });

    ComparisonSimpleEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends,
                                                        ComparisonOperation::Greater,
                                                        expectedOutput);
}

TEST_CASE("ClGreaterBroadcastEndToEndTest")
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ComparisonBroadcastEndToEnd<armnn::DataType::Float32>(clDefaultBackends,
                                                          ComparisonOperation::Greater,
                                                          expectedOutput);
}

TEST_CASE("ClGreaterBroadcastEndToEndUint8Test")
{
    const std::vector<uint8_t> expectedOutput({ 0, 1, 0, 0, 0, 1,
                                                1, 1, 1, 1, 1, 1 });

    ComparisonBroadcastEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends,
                                                           ComparisonOperation::Greater,
                                                           expectedOutput);
}

// InstanceNormalization
TEST_CASE("ClInstanceNormalizationNhwcEndToEndTest1")
{
    InstanceNormalizationNhwcEndToEndTest1(clDefaultBackends);
}

TEST_CASE("ClInstanceNormalizationNchwEndToEndTest1")
{
    InstanceNormalizationNchwEndToEndTest1(clDefaultBackends);
}

TEST_CASE("ClInstanceNormalizationNhwcEndToEndTest2")
{
    InstanceNormalizationNhwcEndToEndTest2(clDefaultBackends);
}

TEST_CASE("ClInstanceNormalizationNchwEndToEndTest2")
{
    InstanceNormalizationNchwEndToEndTest2(clDefaultBackends);
}

// Pooling 2D
// Average Pool 2D
TEST_CASE("ClAvgPool2DEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClAvgPool2DEndtoEndTestFloat16")
{
    AvgPool2dEndToEndFloat16<DataType::Float16>(clDefaultBackends);
}

TEST_CASE("ClAvgPool2DIgnoreValueEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(clDefaultBackends, PaddingMethod::IgnoreValue);
}

// Max Pool 2D
TEST_CASE("ClMaxPool2DEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClMaxPool2DEndtoEndTestFloat16")
{
    MaxPool2dEndToEndFloat16<DataType::Float16>(clDefaultBackends);
}

TEST_CASE("ClMaxPool2DIgnoreValueEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(clDefaultBackends, PaddingMethod::IgnoreValue);
}

TEST_CASE("ClMaxPool2DTwoLayerEndtoEndTestFloat32")
{
    MaxPool2dTwoLayerEndToEnd<DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClMaxPool2DThreeLayerEndtoEndTestFloat32")
{
    MaxPool2dThreeLayerEndToEnd<DataType::Float32>(clDefaultBackends);
}

// Fill
TEST_CASE("ClFillEndToEndTest")
{
    FillEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClFillEndToEndTestFloat16")
{
    FillEndToEnd<armnn::DataType::Float16>(clDefaultBackends);
}

TEST_CASE("ClFillEndToEndTestInt32")
{
    FillEndToEnd<armnn::DataType::Signed32>(clDefaultBackends);
}

// Prelu
TEST_CASE("ClPreluEndToEndFloat32Test")
{
    PreluEndToEndNegativeTest<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClPreluEndToEndTestUint8")
{
    PreluEndToEndPositiveTest<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

// Reduce
// Reduce Sum
TEST_CASE("ClReduceSumSum2dEndtoEndTestSigned32")
{
    ReduceEndToEnd2d<DataType::Signed32>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum2dEndtoEndTestSigned32WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Signed32>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum2dEndtoEndTestFloat16")
{
    ReduceEndToEnd2d<DataType::Float16>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum2dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Float16>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum2dEndtoEndTestFloat32")
{
    ReduceEndToEnd2d<DataType::Float32>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum2dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Float32>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum2dEndtoEndTestInt8")
{
    ReduceEndToEnd2d<DataType::QAsymmS8>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum2dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd2d<DataType::QAsymmS8>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum3dEndtoEndTestSigned32")
{
    ReduceEndToEnd3d<DataType::Signed32>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum3dEndtoEndTestSigned32WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Signed32>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum3dEndtoEndTestFloat16")
{
    ReduceEndToEnd3d<DataType::Float16>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum3dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Float16>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum3dEndtoEndTestFloat32")
{
    ReduceEndToEnd3d<DataType::Float32>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum3dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Float32>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum3dEndtoEndTestInt8")
{
    ReduceEndToEnd3d<DataType::QAsymmS8>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum3dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd3d<DataType::QAsymmS8>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum4dEndtoEndTestSigned32")
{
    ReduceEndToEnd4d<DataType::Signed32>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum4dEndtoEndTestSigned32WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Signed32>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum4dEndtoEndTestFloat16")
{
    ReduceEndToEnd4d<DataType::Float16>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum4dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Float16>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum4dEndtoEndTestFloat32")
{
    ReduceEndToEnd4d<DataType::Float32>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum4dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Float32>(clDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("ClReduceSumSum4dEndtoEndTestInt8")
{
    ReduceEndToEnd4d<DataType::QAsymmS8>(clDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("ClReduceSumSum4dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd4d<DataType::QAsymmS8>(clDefaultBackends, ReduceOperation::Sum, true);
}

// Reshape
TEST_CASE("ClReshapeEndToEndTest")
{
    ReshapeEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClReshapeEndToEndTestFloat16")
{
    ReshapeEndToEndFloat16<armnn::DataType::Float16>(clDefaultBackends);
}

TEST_CASE("ClReshapeEndToEndTestInt32")
{
    ReshapeEndToEnd<armnn::DataType::Signed32>(clDefaultBackends);
}

TEST_CASE("ClReshapeEndToEndTestInt16")
{
    ReshapeEndToEnd<armnn::DataType::QSymmS16>(clDefaultBackends);
}

TEST_CASE("ClReshapeEndToEndTestUInt8")
{
    ReshapeEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClReshapeEndToEndTestInt8")
{
    ReshapeEndToEnd<armnn::DataType::QAsymmS8>(clDefaultBackends);
}

// Resize Bilinear
TEST_CASE("ClResizeBilinearEndToEndFloatNchwTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::Float32>(clDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("ClResizeBilinearEndToEndFloatNhwcTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::Float32>(clDefaultBackends, armnn::DataLayout::NHWC);
}

// Resize NearestNeighbor
TEST_CASE("ClResizeNearestNeighborEndToEndFloatNchwTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(clDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("ClResizeNearestNeighborEndToEndFloatNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(clDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("ClResizeNearestNeighborEndToEndFloatAlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(clDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("ClResizeNearestNeighborEndToEndFloatHalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(clDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

// ReverseV2
TEST_CASE("ClReverseV2EndToEndTest")
{
    ReverseV2EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClReverseV2EndToEndSigned32Test")
{
    ReverseV2EndToEnd<armnn::DataType::Signed32>(clDefaultBackends);
}

TEST_CASE("ClReverseV2EndToEndSigned64Test")
{
    ReverseV2EndToEnd<armnn::DataType::Signed64>(clDefaultBackends);
}

// Space to depth
TEST_CASE("ClSpaceToDepthNhwcEndToEndTest1")
{
    SpaceToDepthNhwcEndToEndTest1(clDefaultBackends);
}

TEST_CASE("ClSpaceToDepthNchwEndToEndTest1")
{
    SpaceToDepthNchwEndToEndTest1(clDefaultBackends);
}

TEST_CASE("ClSpaceToDepthNhwcEndToEndTest2")
{
    SpaceToDepthNhwcEndToEndTest2(clDefaultBackends);
}

TEST_CASE("ClSpaceToDepthNchwEndToEndTest2")
{
    SpaceToDepthNchwEndToEndTest2(clDefaultBackends);
}

// Split
TEST_CASE("ClSplitter1dEndToEndTest")
{
    Splitter1dEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter1dEndToEndUint8Test")
{
    Splitter1dEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClSplitter2dDim0EndToEndTest")
{
    Splitter2dDim0EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter2dDim1EndToEndTest")
{
    Splitter2dDim1EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter2dDim0EndToEndUint8Test")
{
    Splitter2dDim0EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClSplitter2dDim1EndToEndUint8Test")
{
    Splitter2dDim1EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClSplitter3dDim0EndToEndTest")
{
    Splitter3dDim0EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter3dDim1EndToEndTest")
{
    Splitter3dDim1EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter3dDim2EndToEndTest")
{
    Splitter3dDim2EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter3dDim0EndToEndUint8Test")
{
    Splitter3dDim0EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClSplitter3dDim1EndToEndUint8Test")
{
    Splitter3dDim1EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClSplitter3dDim2EndToEndUint8Test")
{
    Splitter3dDim2EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClSplitter4dDim0EndToEndTest")
{
    Splitter4dDim0EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter4dDim1EndToEndTest")
{
    Splitter4dDim1EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter4dDim2EndToEndTest")
{
    Splitter4dDim2EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter4dDim3EndToEndTest")
{
    Splitter4dDim3EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClSplitter4dDim0EndToEndUint8Test")
{
    Splitter4dDim0EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClSplitter4dDim1EndToEndUint8Test")
{
    Splitter4dDim1EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClSplitter4dDim2EndToEndUint8Test")
{
    Splitter4dDim2EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClSplitter4dDim3EndToEndUint8Test")
{
    Splitter4dDim3EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

// Tile
TEST_CASE("ClTileEndToEndFloat32")
{
    TileEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClTileEndToEndFloat16")
{
    TileEndToEnd<armnn::DataType::Float16>(clDefaultBackends);
}

TEST_CASE("ClTileEndToEndQAsymmS8")
{
    TileEndToEnd<armnn::DataType::QAsymmS8>(clDefaultBackends);
}

TEST_CASE("ClTileEndToEndQAsymmU8")
{
    TileEndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClTileEndToEndQSymmS8")
{
    TileEndToEnd<armnn::DataType::QSymmS8>(clDefaultBackends);
}

TEST_CASE("ClTileEndToEndQSymmS16")
{
    TileEndToEnd<armnn::DataType::QSymmS16>(clDefaultBackends);
}

TEST_CASE("ClTileEndToEndSigned32")
{
    TileEndToEnd<armnn::DataType::Signed32>(clDefaultBackends);
}

// TransposeConvolution2d
TEST_CASE("ClTransposeConvolution2dEndToEndFloatNchwTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        clDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("ClTransposeConvolution2dEndToEndUint8NchwTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        clDefaultBackends, armnn::DataLayout::NCHW);
}

TEST_CASE("ClTransposeConvolution2dEndToEndFloatNhwcTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        clDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("ClTransposeConvolution2dEndToEndUint8NhwcTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::QAsymmU8, armnn::DataType::Signed32>(
        clDefaultBackends, armnn::DataLayout::NHWC);
}

// Transpose
TEST_CASE("ClTransposeEndToEndTest")
{
TransposeEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClQuantizedLstmEndToEndTest")
{
    QuantizedLstmEndToEnd(clDefaultBackends);
}

// ArgMinMax
TEST_CASE("ClArgMaxSimpleTest")
{
    ArgMaxEndToEndSimple<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMinSimpleTest")
{
    ArgMinEndToEndSimple<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMaxAxis0Test")
{
    ArgMaxAxis0EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMinAxis0Test")
{
    ArgMinAxis0EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMaxAxis1Test")
{
    ArgMaxAxis1EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMinAxis1Test")
{
    ArgMinAxis1EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMaxAxis2Test")
{
    ArgMaxAxis2EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMinAxis2Test")
{
    ArgMinAxis2EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMaxAxis3Test")
{
    ArgMaxAxis3EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMinAxis3Test")
{
    ArgMinAxis3EndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClArgMaxSimpleTestQAsymmU8")
{
    ArgMaxEndToEndSimple<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClArgMinSimpleTestQAsymmU8")
{
    ArgMinEndToEndSimple<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClArgMaxAxis0TestQAsymmU8")
{
    ArgMaxAxis0EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClArgMinAxis0TestQAsymmU8")
{
    ArgMinAxis0EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClArgMaxAxis1TestQAsymmU8")
{
    ArgMaxAxis1EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClArgMinAxis1TestQAsymmU8")
{
    ArgMinAxis1EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClArgMaxAxis2TestQAsymmU8")
{
    ArgMaxAxis2EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClArgMinAxis2TestQAsymmU8")
{
    ArgMinAxis2EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClArgMaxAxis3TestQAsymmU8")
{
    ArgMaxAxis3EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClArgMinAxis3TestQAsymmU8")
{
    ArgMinAxis3EndToEnd<armnn::DataType::QAsymmU8>(clDefaultBackends);
}

TEST_CASE("ClQLstmEndToEndTest")
{
    QLstmEndToEnd(clDefaultBackends);
}

TEST_CASE("ClForceImportWithMisalignedInputBuffersEndToEndTest"
          // Currently, the CL workload for activation does not support tensor handle replacement so this test case
          // will always fail.
          * doctest::skip(true))
{
    ForceImportWithMisalignedInputBuffersEndToEndTest(clDefaultBackends);
}

TEST_CASE("ClForceImportWithMisalignedOutputBuffersEndToEndTest"
          // Currently, the CL workload for activation does not support tensor handle replacement so this test case
          // will always fail.
          * doctest::skip(true))
{
    ForceImportWithMisalignedOutputBuffersEndToEndTest(clDefaultBackends);
}

TEST_CASE("ClForceImportWithMisalignedInputAndOutputBuffersEndToEndTest")
{
    ForceImportWithMisalignedInputAndOutputBuffersEndToEndTest(clDefaultBackends);
}

TEST_CASE("ClReshapeRemovalSimpleCaseEndToEnd")
{
    ReshapeRemovalEndToEnd<armnn::DataType::Float32>(clDefaultBackends);
}

TEST_CASE("ClReshapeRemovalNCHWFirstEndToEnd")
{
    ReshapeRemovalNCHWEndToEnd<armnn::DataType::Float32>(clDefaultBackends, false, true);
}

TEST_CASE("ClReshapeRemovalNCHWSecondEndToEnd")
{
    ReshapeRemovalNCHWEndToEnd<armnn::DataType::Float32>(clDefaultBackends, false, false);
}

}
