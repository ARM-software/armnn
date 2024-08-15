//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/test/EndToEndTestImpl.hpp"

#include "backendsCommon/test/ActivationEndToEndTestImpl.hpp"
#include "backendsCommon/test/AdditionEndToEndTestImpl.hpp"
#include "backendsCommon/test/BatchMatMulEndToEndTestImpl.hpp"
#include "backendsCommon/test/Convolution2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/ConcatEndToEndTestImpl.hpp"
#include "backendsCommon/test/DepthwiseConvolution2dEndToEndTests.hpp"
#include "backendsCommon/test/ElementwiseBinaryEndToEndTestImpl.hpp"
#include "backendsCommon/test/ElementwiseUnaryEndToEndTestImpl.hpp"
#include "backendsCommon/test/FullyConnectedEndToEndTestImpl.hpp"
#include "backendsCommon/test/MeanEndToEndTestImpl.hpp"
#include "backendsCommon/test/MultiplicationEndToEndTestImpl.hpp"
#include "backendsCommon/test/PadEndToEndTestImpl.hpp"
#include "backendsCommon/test/Pooling2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/QuantizationEndToEndTestImpl.hpp"
#include "backendsCommon/test/ReduceEndToEndTestImpl.hpp"
#include "backendsCommon/test/ReshapeEndToEndTestImpl.hpp"
#include "backendsCommon/test/ResizeEndToEndTestImpl.hpp"
#include "backendsCommon/test/SliceEndToEndTestImpl.hpp"
#include "backendsCommon/test/SoftmaxEndToEndTestImpl.hpp"
#include "backendsCommon/test/SplitterEndToEndTestImpl.hpp"
#include "backendsCommon/test/SubtractionEndToEndTestImpl.hpp"
#include "backendsCommon/test/TransposeConvolution2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/TransposeEndToEndTestImpl.hpp"

#include <doctest/doctest.h>

TEST_SUITE("TosaRefEndToEnd")
{
static std::vector<BackendId> tosaDefaultBackends = { "TosaRef" };

// Activation
// LeakyRelu
TEST_CASE("TosaRefLeakyReluActivationFloat32")
{
    ActivationEndToEndTest<DataType::Float32>(tosaDefaultBackends, ActivationFunction::LeakyReLu, 1.f, 0, 0.01f);
}

TEST_CASE("TosaRefLeakyReluActivationFloat16")
{
    ActivationEndToEndTest<DataType::Float16>(tosaDefaultBackends, ActivationFunction::LeakyReLu, 0.3f, 5, 0.01f);
}

TEST_CASE("TosaRefLeakyReluActivationInt32")
{
    ActivationEndToEndTest<DataType::Signed32>(tosaDefaultBackends, ActivationFunction::LeakyReLu, 0.15f, 0, 0.01f);
}

TEST_CASE("TosaRefLeakyReluActivationInt16")
{
    ActivationEndToEndTest<DataType::QSymmS16>(tosaDefaultBackends, ActivationFunction::LeakyReLu, 0.35f, 0, 0.01f);
}

TEST_CASE("TosaRefLeakyReluActivationInt8")
{
    ActivationEndToEndTest<DataType::QAsymmS8>(tosaDefaultBackends, ActivationFunction::LeakyReLu, 0.6f, 7, 0.01f);
}

TEST_CASE("UNSUPPORTED_ActivationUInt8")
{
    try
    {
        ActivationEndToEndTest<DataType::QAsymmU8>(tosaDefaultBackends, ActivationFunction::LeakyReLu, 1.f, 0, 0.01f);
        FAIL("An exception should have been thrown");
    }
    catch (armnn::Exception& e)
    {
        CHECK_EQ(std::string(e.what()), "Failed to assign a backend to each layer");
    }
}

// Relu
TEST_CASE("TosaRefReLuEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(tosaDefaultBackends, ActivationFunction::ReLu);
}

TEST_CASE("TosaRefReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(tosaDefaultBackends, ActivationFunction::ReLu);
}

TEST_CASE("TosaRefReLuEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(tosaDefaultBackends, ActivationFunction::ReLu);
}

TEST_CASE("TosaRefReLuEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<armnn::DataType::QSymmS16>(tosaDefaultBackends, ActivationFunction::ReLu);
}

// Gelu
TEST_CASE("TosaRefGeluEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(tosaDefaultBackends, ActivationFunction::Gelu);
}

// Sigmoid
TEST_CASE("TosaRefSigmoidEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(tosaDefaultBackends, ActivationFunction::Sigmoid);
}

TEST_CASE("TosaRefSigmoidEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(tosaDefaultBackends, ActivationFunction::Sigmoid);
}

TEST_CASE("TosaRefSigmoidEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(tosaDefaultBackends, ActivationFunction::Sigmoid);
}

// TanH
TEST_CASE("TosaRefTanHEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(tosaDefaultBackends, ActivationFunction::TanH);
}

// BoundedRelu
TEST_CASE("TosaRefBoundedReLuEndToEndTestFloat32")
{
    ActivationEndToEndTest<armnn::DataType::Float32>(
        tosaDefaultBackends, ActivationFunction::BoundedReLu, 1.0f, 0, 6.0f, 0.0f);
}

TEST_CASE("TosaRefBoundedReLuEndToEndTestFloat16")
{
    ActivationEndToEndTest<armnn::DataType::Float16>(
        tosaDefaultBackends, ActivationFunction::BoundedReLu, 1.0f, 0, 6.0f, 0.0f);
}

TEST_CASE("TosaRefBoundedReLuEndToEndTestQAsymmS8")
{
    ActivationEndToEndTest<armnn::DataType::QAsymmS8>(
        tosaDefaultBackends, ActivationFunction::BoundedReLu, 1.0f, 0, 6.0f, 0.0f);
}

TEST_CASE("TosaRefBoundedReLuEndToEndTestQSymmS16")
{
    ActivationEndToEndTest<armnn::DataType::QSymmS16>(
        tosaDefaultBackends, ActivationFunction::BoundedReLu, 1.0f, 0, 6.0f, 0.0f);
}

// Addition
TEST_CASE("TosaRefAdditionEndtoEndTestFloat32")
{
    AdditionEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefAdditionEndtoEndTestInt32")
{
    AdditionEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefAdditionEndtoEndTestFloat16")
{
    AdditionEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

// BatchMatMul

TEST_CASE("TosaRefBatchMatMulEndToEndFloat32Test")
{
    BatchMatMulEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulEndToEndInt8Test")
{
    BatchMatMulEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulEndToEndInt16Test")
{
    BatchMatMulEndToEnd<armnn::DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulNoTransposeEndToEndFloat32Test")
{
    BatchMatMulNoTransposeEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulNoTransposeEndToEndInt8Test")
{
    BatchMatMulNoTransposeEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulNoTransposeEndToEndInt16Test")
{
    BatchMatMulNoTransposeEndToEnd<armnn::DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulSimple4DEndToEndFloat32Test")
{
    BatchMatMulSimple4DEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulSimple4DEndToEndInt8Test")
{
    BatchMatMulSimple4DEndToEnd<armnn::DataType::QAsymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulSimple4DEndToEndInt16Test")
{
    BatchMatMulSimple4DEndToEnd<armnn::DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulNotSquareEndToEndFloat32Test")
{
    BatchMatMulNotSquareEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulNotSquareEndToEndInt8Test")
{
    BatchMatMulNotSquareEndToEnd<armnn::DataType::QAsymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMulNotSquareEndToEndInt16Test")
{
    BatchMatMulNotSquareEndToEnd<armnn::DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMul4DEndToEndFloat32Test")
{
    BatchMatMul4DEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMul4DEndToEndInt8Test")
{
    BatchMatMul4DEndToEnd<armnn::DataType::QAsymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefBatchMatMul4DEndToEndInt16Test")
{
    BatchMatMul4DEndToEnd<armnn::DataType::QSymmS16>(tosaDefaultBackends);
}

// Concat
TEST_CASE("TosaRefConcatEndToEndDim0TestFloat32")
{
    ConcatDim0EndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim0TestInt32")
{
    ConcatDim0EndToEnd<armnn::DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim1TestFloat32")
{
    ConcatDim1EndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim1TestInt32")
{
    ConcatDim1EndToEnd<armnn::DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim2TestFloat32")
{
    ConcatDim2EndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim2TestInt32")
{
    ConcatDim2EndToEnd<armnn::DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim3TestFloat32")
{
    ConcatDim3EndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefConcatEndToEndDim3TestInt32")
{
    ConcatDim3EndToEnd<armnn::DataType::Signed32>(tosaDefaultBackends);
}

// Conv2d
TEST_CASE("TosaRefConv2dEndtoEndTestFloat32")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("TosaRefConv2dWithoutBiasEndtoEndTestFloat32")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC, false);
}

TEST_CASE("TosaRefConv2dEndtoEndTestInt8")
{
    Convolution2dEndToEnd<armnn::DataType::QSymmS8,
                          armnn::DataType::QSymmS8,
                          armnn::DataType::Signed32>(tosaDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("TosaRefConv2dWithoutBiasEndtoEndTestInt8")
{
    Convolution2dEndToEnd<armnn::DataType::QSymmS8,
                          armnn::DataType::QSymmS8,
                          armnn::DataType::Signed32>(tosaDefaultBackends, armnn::DataLayout::NHWC, false);
}

// DepthwiseConv2d
TEST_CASE("TosaRefDepthwiseConv2dEndtoEndTestInt8")
{
    DepthwiseConvolution2dEndToEnd<armnn::DataType::QSymmS8,
                                   armnn::DataType::Signed32>(tosaDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("TosaRefDepthwiseConv2dEndtoEndTestInt8BiasDisabled")
{
    DepthwiseConvolution2dEndToEnd<armnn::DataType::QSymmS8,
                                   armnn::DataType::Signed32>(tosaDefaultBackends, armnn::DataLayout::NHWC, false);
}

// Elementwise Binary
//Add
TEST_CASE("TosaRefAddEndtoEndTestInt32")
{
    ElementwiseBinarySimpleNoReshapeEndToEnd<DataType::Signed32>(tosaDefaultBackends,
                                                                 armnn::BinaryOperation::Add);
}

TEST_CASE("TosaRefAddEndtoEndTestInt8")
{
    ElementwiseBinarySimpleNoReshapeEndToEnd<DataType::QSymmS8>(tosaDefaultBackends,
                                                                armnn::BinaryOperation::Add);
}

// Maximum
TEST_CASE("TosaRefMaximumEndtoEndTestInt32")
{
    ElementwiseBinarySimpleNoReshapeEndToEnd<DataType::Signed32>(tosaDefaultBackends,
                                                                 armnn::BinaryOperation::Maximum);
}

TEST_CASE("TosaRefMaximumEndtoEndTestInt8")
{
    ElementwiseBinarySimpleNoReshapeEndToEnd<DataType::QSymmS8>(tosaDefaultBackends,
                                                                 armnn::BinaryOperation::Maximum);
}

//Mul
TEST_CASE("TosaRefMulEndtoEndTestInt32")
{
    ElementwiseBinarySimpleNoReshapeEndToEnd<DataType::Signed32>(tosaDefaultBackends,
                                                                 armnn::BinaryOperation::Mul);
}

TEST_CASE("TosaRefMulEndtoEndTestInt8")
{
    ElementwiseBinarySimpleNoReshapeEndToEnd<DataType::QSymmS8>(tosaDefaultBackends,
                                                                armnn::BinaryOperation::Mul);
}

//Sub
TEST_CASE("TosaRefMulEndtoEndTestInt32")
{
    ElementwiseBinarySimpleNoReshapeEndToEnd<DataType::Signed32>(tosaDefaultBackends,
                                                                 armnn::BinaryOperation::Sub);
}

TEST_CASE("TosaRefSubEndtoEndTestInt8")
{
    ElementwiseBinarySimpleNoReshapeEndToEnd<DataType::QSymmS8>(tosaDefaultBackends,
                                                                armnn::BinaryOperation::Sub);
}

// FullyConnected
TEST_CASE("TosaRefFullyConnectedEndToEndTestFloat32")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, true);
}

TEST_CASE("TosaRefFullyConnectedEndToEndTestNoBiasFloat32")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, false);
}

TEST_CASE("TosaRefFullyConnectedEndToEndTestInt8")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QAsymmS8,
                                                 armnn::DataType::QAsymmS8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QAsymmS8>(tosaDefaultBackends, true);
}

TEST_CASE("TosaRefFullyConnectedEndToEndTestNoBiasInt8")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QAsymmS8,
                                                 armnn::DataType::QAsymmS8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QAsymmS8>(tosaDefaultBackends, false);
}

TEST_CASE("TosaRefFullyConnectedEndToEndTestInt8Symm")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QSymmS8,
                                                 armnn::DataType::QSymmS8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QSymmS8>(tosaDefaultBackends, true);
}

TEST_CASE("TosaRefFullyConnectedEndToEndTestNoBiasInt8Symm")
{
    FullyConnectedConstantWeightsAndBiasEndToEnd<armnn::DataType::QSymmS8,
                                                 armnn::DataType::QSymmS8,
                                                 armnn::DataType::Signed32,
                                                 armnn::DataType::QSymmS8>(tosaDefaultBackends, false);
}

// Pad
TEST_CASE("TosaRefPadEndToEndFloat32Test")
{
    PadEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefPadEndToEndInt8Test")
{
    PadEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefPad4dEndToEndFloat32Test")
{
    Pad4dEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefPad4dEndToEndInt8Test")
{
    Pad4dEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefPad4dEndToEndInt32Test")
{
    Pad4dEndToEnd<armnn::DataType::Signed32>(tosaDefaultBackends);
}

// Pooling
// Average Pool 2D
TEST_CASE("TosaRefAvgPool2DEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefAvgPool2DEndtoEndTestFloat16")
{
    AvgPool2dEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefAvgPool2DIgnoreValueEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(tosaDefaultBackends, PaddingMethod::IgnoreValue);
}

// Max Pool 2D
TEST_CASE("TosaRefMaxPool2DEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMaxPool2DEndtoEndTestFloat16")
{
    MaxPool2dEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMaxPool2DIgnoreValueEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(tosaDefaultBackends, PaddingMethod::IgnoreValue);
}

TEST_CASE("TosaRefMaxPool2DTwoLayerEndtoEndTestFloat32")
{
    MaxPool2dTwoLayerEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMaxPool2DThreeLayerEndtoEndTestFloat32")
{
    MaxPool2dThreeLayerEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

// Mean
TEST_CASE("TosaRefMeanEndToEndFloat16Test")
{
    MeanEndToEnd<armnn::DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMeanEndToEndFloat16TestWithKeepDims")
{
    MeanEndToEnd<armnn::DataType::Float16>(tosaDefaultBackends, true);
}

TEST_CASE("TosaRefMeanEndToEndFloat32Test")
{
    MeanEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMeanEndToEndFloat32TestWithKeepDims")
{
    MeanEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, true);
}

TEST_CASE("TosaRefMeanEndToEndInt8Test")
{
    MeanEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMeanEndToEndInt8TestWithKeepDims")
{
    MeanEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends, true);
}

// Quantization
TEST_CASE("TosaRefQuantizeFromFloat32ToInt8")
{
    QuantizationEndToEndFloat32<DataType::QAsymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat32ToInt16")
{
    QuantizationEndToEndFloat32<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat32ToInt32")
{
    QuantizationEndToEndFloat32<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat16ToInt8")
{
    QuantizationEndToEndFloat16<DataType::QAsymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat16ToInt16")
{
    QuantizationEndToEndFloat16<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefQuantizeFromFloat16ToInt32")
{
    QuantizationEndToEndFloat16<DataType::Signed32>(tosaDefaultBackends);
}

// Reduce
// Reduce Sum
TEST_CASE("TosaRefReduceSum2dEndtoEndTestSigned32")
{
    ReduceEndToEnd2d<DataType::Signed32>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum2dEndtoEndTestSigned32WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Signed32>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum2dEndtoEndTestFloat16")
{
    ReduceEndToEnd2d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum2dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum2dEndtoEndTestFloat32")
{
    ReduceEndToEnd2d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum2dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum2dEndtoEndTestInt8")
{
    ReduceEndToEnd2d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum2dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd2d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum3dEndtoEndTestSigned32")
{
    ReduceEndToEnd3d<DataType::Signed32>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum3dEndtoEndTestSigned32WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Signed32>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum3dEndtoEndTestFloat16")
{
    ReduceEndToEnd3d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum3dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum3dEndtoEndTestFloat32")
{
    ReduceEndToEnd3d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum3dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum3dEndtoEndTestInt8")
{
    ReduceEndToEnd3d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum3dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd3d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum4dEndtoEndTestSigned32")
{
    ReduceEndToEnd4d<DataType::Signed32>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum4dEndtoEndTestSigned32WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Signed32>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum4dEndtoEndTestFloat16")
{
    ReduceEndToEnd4d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum4dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum4dEndtoEndTestFloat32")
{
    ReduceEndToEnd4d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum4dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

TEST_CASE("TosaRefReduceSum4dEndtoEndTestInt8")
{
    ReduceEndToEnd4d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Sum);
}

TEST_CASE("TosaRefReduceSum4dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd4d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Sum, true);
}

// Reduce Mean
TEST_CASE("TosaRefReduceMean2dEndtoEndTestFloat16")
{
    ReduceEndToEnd2d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Mean);
}

TEST_CASE("TosaRefReduceMean2dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Mean, true);
}

TEST_CASE("TosaRefReduceMean2dEndtoEndTestFloat32")
{
    ReduceEndToEnd2d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Mean);
}

TEST_CASE("TosaRefReduceMean2dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd2d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Mean, true);
}

TEST_CASE("TosaRefReduceMean2dEndtoEndTestInt8")
{
    ReduceEndToEnd2d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Mean);
}

TEST_CASE("TosaRefReduceMean2dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd2d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Mean, true);
}

TEST_CASE("TosaRefReduceMean3dEndtoEndTestFloat16")
{
    ReduceEndToEnd3d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Mean);
}

TEST_CASE("TosaRefReduceMean3dEndtoEndTestFloat16WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Float16>(tosaDefaultBackends, ReduceOperation::Mean, true);
}

TEST_CASE("TosaRefReduceMean3dEndtoEndTestFloat32")
{
    ReduceEndToEnd3d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Mean);
}

TEST_CASE("TosaRefReduceMean3dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd3d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Mean, true);
}

TEST_CASE("TosaRefReduceMean3dEndtoEndTestInt8")
{
    ReduceEndToEnd3d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Mean);
}

TEST_CASE("TosaRefReduceMean3dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd3d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Mean, true);
}

TEST_CASE("TosaRefReduceMean4dEndtoEndTestFloat32")
{
    ReduceEndToEnd4d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Mean);
}

TEST_CASE("TosaRefReduceMean4dEndtoEndTestFloat32WithKeepDims")
{
    ReduceEndToEnd4d<DataType::Float32>(tosaDefaultBackends, ReduceOperation::Mean, true);
}

TEST_CASE("TosaRefReduceMean4dEndtoEndTestInt8")
{
    ReduceEndToEnd4d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Mean);
}

TEST_CASE("TosaRefReduceMean4dEndtoEndTestInt8WithKeepDims")
{
    ReduceEndToEnd4d<DataType::QAsymmS8>(tosaDefaultBackends, ReduceOperation::Mean, true);
}

// Reshape
TEST_CASE("TosaRefReshapeEndtoEndTestFloat32")
{
    ReshapeEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefReshapeEndtoEndTestInt32")
{
    ReshapeEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefReshapeEndtoEndTestFloat16")
{
    ReshapeEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

// Rsqrt
TEST_CASE("TosaRefRsqrtEndtoEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<DataType::Float32>(tosaDefaultBackends,
                                                      UnaryOperation::Rsqrt);
}

TEST_CASE("TosaRefRsqrtEndtoEndTestFloat16")
{
    ElementwiseUnarySimpleEndToEnd<DataType::Float16>(tosaDefaultBackends,
                                                      UnaryOperation::Rsqrt);
}

TEST_CASE("TosaRefRsqrtEndToEndTestInt8")
{
    ElementwiseUnarySimpleEndToEnd<DataType::QSymmS8>(tosaDefaultBackends,
                                                      UnaryOperation::Rsqrt);
}

TEST_CASE("TosaRefRsqrtEndToEndTestQAsymmS8")
{
    ElementwiseUnarySimpleEndToEnd<DataType::QAsymmS8>(tosaDefaultBackends,
                                                       UnaryOperation::Rsqrt);
}

// Exp
TEST_CASE("TosaRefExpEndToEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends,
                                                             UnaryOperation::Exp);
}

TEST_CASE("TosaRefExpEndToEndTestInt8")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::QAsymmS8>(tosaDefaultBackends,
                                                              UnaryOperation::Exp);
}

// Log
TEST_CASE("TosaRefLogEndToEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends,
                                                             UnaryOperation::Log);
}

TEST_CASE("TosaRefLogEndToEndTestSint8")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::QAsymmS8>(tosaDefaultBackends,
                                                              UnaryOperation::Log);
}

// Resize
TEST_CASE("TosaRefResizeNearestNeighborEndToEndFloat32AlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndFloat32HalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndFloat16AlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float16>(tosaDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndFloat16HalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float16>(tosaDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndInt8AlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndInt8HalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS8>(tosaDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndInt16AlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS16>(tosaDefaultBackends, armnn::DataLayout::NHWC, true, false);
}

TEST_CASE("TosaRefResizeNearestNeighborEndToEndInt16HalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::QSymmS16>(tosaDefaultBackends, armnn::DataLayout::NHWC, false, true);
}

// Slice
TEST_CASE("TosaRefSliceEndtoEndTestFloat32")
{
    SliceEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSliceEndtoEndTestInt32")
{
    SliceEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSliceEndtoEndTestFloat16")
{
    SliceEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

// Softmax
TEST_CASE("TosaRef3DSoftmaxQuantizedInt8")
{
    QSoftmax3DEndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRef1DSoftmaxQuantizedInt8")
{
    QSoftmax1DEndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

// Split
TEST_CASE("TosaRefSplit1dEndtoEndTestBoolean")
{
    Splitter1dEndToEnd<DataType::Boolean>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndtoEndTestInt8")
{
    Splitter1dEndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndtoEndTestSigned16")
{
    Splitter1dEndToEnd<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndtoEndTestInt32")
{
    Splitter1dEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndtoEndTestFloat16")
{
    Splitter1dEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit1dEndToEndFloat32")
{
    Splitter1dEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit2dDim0EndtoEndTestFloat32")
{
    Splitter2dDim0EndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit2dDim1EndtoEndTestFloat32")
{
    Splitter2dDim1EndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim0EndtoEndTestFloat32")
{
    Splitter3dDim0EndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestFloat32")
{
    Splitter3dDim1EndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestFloat16")
{
    Splitter3dDim1EndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestBoolean")
{
    Splitter3dDim1EndToEnd<DataType::Boolean>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestInt8")
{
    Splitter3dDim1EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestSigned16")
{
    Splitter3dDim1EndToEnd<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim1EndtoEndTestInt32")
{
    Splitter3dDim1EndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit3dDim2EndtoEndTestInt8")
{
    Splitter3dDim2EndToEnd<DataType::QAsymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim0EndtoEndTestInt8")
{
    Splitter4dDim0EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim1EndtoEndTestInt8")
{
    Splitter4dDim1EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestBoolean")
{
    Splitter4dDim2EndToEnd<DataType::Boolean>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestInt8")
{
    Splitter4dDim2EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestInt16")
{
    Splitter4dDim2EndToEnd<DataType::QSymmS16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestInt32")
{
    Splitter4dDim2EndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim2EndtoEndTestFloat16")
{
    Splitter4dDim2EndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSplit4dDim3EndtoEndTestInt8")
{
    Splitter4dDim3EndToEnd<DataType::QSymmS8>(tosaDefaultBackends);
}

// Subtraction
TEST_CASE("TosaRefSubtractionEndtoEndTestFloat32")
{
    SubtractionEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSubtractionEndtoEndTestInt32")
{
    SubtractionEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefSubtractionEndtoEndTestFloat16")
{
    SubtractionEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMultiplicationEndtoEndTestFloat32")
{
    MultiplicationEndToEnd<DataType::Float32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMultiplicationEndtoEndTestInt32")
{
    MultiplicationEndToEnd<DataType::Signed32>(tosaDefaultBackends);
}

TEST_CASE("TosaRefMultiplicationEndtoEndTestFloat16")
{
    MultiplicationEndToEndFloat16<DataType::Float16>(tosaDefaultBackends);
}

// TransposeConvolution2d
TEST_CASE("TosaRefTransposeConvolution2dEndToEndFloatNhwcTest")
{
    TransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        tosaDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("TosaRefSimpleTransposeConvolution2dEndToEndFloatNhwcTest")
{
    SimpleTransposeConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(
        tosaDefaultBackends, armnn::DataLayout::NHWC);
}

// Transpose
TEST_CASE("TosaRefTransposeEndtoEndTestFloat32")
{
    TransposeEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends);
}

}