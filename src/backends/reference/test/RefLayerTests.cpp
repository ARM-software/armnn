//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefWorkloadFactoryHelper.hpp"

#include <test/TensorHelpers.hpp>
#include <test/UnitTests.hpp>

#include <reference/RefWorkloadFactory.hpp>
#include <backendsCommon/test/DetectionPostProcessLayerTestImpl.hpp>
#include <backendsCommon/test/LayerTests.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Compute_Reference)
using FactoryType = armnn::RefWorkloadFactory;

// ============================================================================
// UNIT tests

// Convolution
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x5, SimpleConvolution2d3x5Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x5Uint8, SimpleConvolution2d3x5Uint8Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x5Nhwc, SimpleConvolution2d3x5Test, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x5Uint8Nhwc, SimpleConvolution2d3x5Uint8Test, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x5QSymm16, SimpleConvolution2d3x5QSymm16Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x5QSymm16Nhwc, SimpleConvolution2d3x5QSymm16Test, true,
                     armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2d, SimpleConvolution2d3x5Test, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolutionUint8, SimpleConvolution2d3x5Uint8Test, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dNhwc, SimpleConvolution2d3x5Test, false, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolutionUint8Nhwc, SimpleConvolution2d3x5Uint8Test, false, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(SimpleConvolution1d, Convolution1dTest, true)
ARMNN_AUTO_TEST_CASE(SimpleConvolution1dUint8, Convolution1dUint8Test, true)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x3, SimpleConvolution2d3x3Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x3Uint8, SimpleConvolution2d3x3Uint8Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x3QSymm16, SimpleConvolution2d3x3QSymm16Test, true, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x3Nhwc, SimpleConvolution2d3x3Test, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x3Uint8Nhwc, SimpleConvolution2d3x3Uint8Test, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x3QSymm16Nhwc, SimpleConvolution2d3x3QSymm16Test, true,
                     armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dSquare, SimpleConvolution2d3x3Test, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dSquareNhwc, SimpleConvolution2d3x3Test, false, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dSquareStride2x2Nhwc,
                     SimpleConvolution2d3x3Stride2x2Test,
                     false,
                     armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2dAsymmetricPaddingLargerThanHalfKernelSize,
                     Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2dAsymmetricPadding, Convolution2dAsymmetricPaddingTest, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2dAsymmetricPaddingLargerThanHalfKernelSizeNhwc,
                     Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2dAsymmetricPaddingNhwc,
                     Convolution2dAsymmetricPaddingTest,
                     armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2dSquareNhwc, SimpleConvolution2d3x3NhwcTest, false)

// Depthwise Convolution
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2d, DepthwiseConvolution2dTest, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dUint8, DepthwiseConvolution2dUint8Test, true, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2d, DepthwiseConvolution2dTest, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dUint8,
                     DepthwiseConvolution2dUint8Test,
                     false,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dQSymm16, DepthwiseConvolution2dInt16Test, true, armnn::DataLayout::NCHW)

// NHWC Depthwise Convolution
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dNhwc, DepthwiseConvolution2dTest, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dUint8Nhwc, DepthwiseConvolution2dUint8Test, true, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dNhwc, DepthwiseConvolution2dTest, false, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dUint8Nhwc,
                     DepthwiseConvolution2dUint8Test,
                     false,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleDepthwiseConvolution2d3x3Dilation3x3Nhwc,
                     SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest)


ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1Int16,
                     DepthwiseConvolution2dDepthMul1Int16Test, true, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test, false, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1Nhwc,
                     DepthwiseConvolution2dDepthMul1Test, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test, true, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1Nhwc,
                     DepthwiseConvolution2dDepthMul1Test, false, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test, false, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dAsymmetric,
                     DepthwiseConvolution2dAsymmetricTest, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dAsymmetric,
                     DepthwiseConvolution2dAsymmetricTest, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dAsymmetricNhwc,
                     DepthwiseConvolution2dAsymmetricTest, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dAsymmetricNhwc,
                     DepthwiseConvolution2dAsymmetricTest, false, armnn::DataLayout::NHWC)


// Pooling
//MaxPooling
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize2x2Stride2x2, SimpleMaxPooling2dSize2x2Stride2x2Test, false)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize2x2Stride2x2Uint8, SimpleMaxPooling2dSize2x2Stride2x2Uint8Test, false)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize2x2Stride2x2Int16, SimpleMaxPooling2dSize2x2Stride2x2Int16Test, false)

ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4, SimpleMaxPooling2dSize3x3Stride2x4Test, false)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4Uint8, SimpleMaxPooling2dSize3x3Stride2x4Uint8Test, false)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4Int16, SimpleMaxPooling2dSize3x3Stride2x4Int16Test, false)

ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2d, SimpleMaxPooling2dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dNhwc, SimpleMaxPooling2dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dUint8, SimpleMaxPooling2dUint8Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dInt16, SimpleMaxPooling2dInt16Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dUint8Nhwc, SimpleMaxPooling2dUint8Test, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dInt16Nhwc, SimpleMaxPooling2dInt16Test, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleMaxPooling2d, IgnorePaddingSimpleMaxPooling2dTest)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleMaxPooling2dUint8, IgnorePaddingSimpleMaxPooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleMaxPooling2dInt16, IgnorePaddingSimpleMaxPooling2dInt16Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingMaxPooling2dSize3, IgnorePaddingMaxPooling2dSize3Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingMaxPooling2dSize3Uint8, IgnorePaddingMaxPooling2dSize3Uint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingMaxPooling2dSize3Int16, IgnorePaddingMaxPooling2dSize3Int16Test)

//AveragePooling
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2d, SimpleAveragePooling2dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dNhwc, SimpleAveragePooling2dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dUint8, SimpleAveragePooling2dUint8Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dInt16, SimpleAveragePooling2dInt16Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dUint8Nhwc, SimpleAveragePooling2dUint8Test, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dInt16Nhwc, SimpleAveragePooling2dInt16Test, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2d, IgnorePaddingSimpleAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2dUint8, IgnorePaddingSimpleAveragePooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2dInt16, IgnorePaddingSimpleAveragePooling2dInt16Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2dNoPadding, IgnorePaddingSimpleAveragePooling2dNoPaddingTest)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2dNoPaddingUint8,
                     IgnorePaddingSimpleAveragePooling2dNoPaddingUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2dNoPaddingInt16,
                     IgnorePaddingSimpleAveragePooling2dNoPaddingInt16Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3, IgnorePaddingAveragePooling2dSize3Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3Uint8, IgnorePaddingAveragePooling2dSize3Uint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3Int16, IgnorePaddingAveragePooling2dSize3Int16Test)

ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3x2Stride2x2,
                     IgnorePaddingAveragePooling2dSize3x2Stride2x2Test, false)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3x2Stride2x2NoPadding,
                     IgnorePaddingAveragePooling2dSize3x2Stride2x2Test, true)

ARMNN_AUTO_TEST_CASE(LargeTensorsAveragePooling2d, LargeTensorsAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE(LargeTensorsAveragePooling2dUint8, LargeTensorsAveragePooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(LargeTensorsAveragePooling2dInt16, LargeTensorsAveragePooling2dInt16Test)

//L2Pooling
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleL2Pooling2d, IgnorePaddingSimpleL2Pooling2dTest)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleL2Pooling2dUint8, IgnorePaddingSimpleL2Pooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleL2Pooling2dInt16, IgnorePaddingSimpleL2Pooling2dInt16Test)

ARMNN_AUTO_TEST_CASE(IgnorePaddingL2Pooling2dSize3, IgnorePaddingL2Pooling2dSize3Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingL2Pooling2dSize3Uint8, IgnorePaddingL2Pooling2dSize3Uint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingL2Pooling2dSize3Int16, IgnorePaddingL2Pooling2dSize3Int16Test)

ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2d, SimpleL2Pooling2dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dNhwc, SimpleL2Pooling2dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dUint8, SimpleL2Pooling2dUint8Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dInt16, SimpleL2Pooling2dInt16Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dNhwcUint8, SimpleL2Pooling2dUint8Test, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dNhwcInt16, SimpleL2Pooling2dInt16Test, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(L2Pooling2dSize7, L2Pooling2dSize7Test)
ARMNN_AUTO_TEST_CASE(L2Pooling2dSize7Uint8, L2Pooling2dSize7Uint8Test)
ARMNN_AUTO_TEST_CASE(L2Pooling2dSize7Int16, L2Pooling2dSize7Int16Test)

//NonSquarePooling
ARMNN_AUTO_TEST_CASE(AsymmNonSquarePooling2d, AsymmetricNonSquarePooling2dTest)
ARMNN_AUTO_TEST_CASE(AsymmNonSquarePooling2dUint8, AsymmetricNonSquarePooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(AsymmNonSquarePooling2dInt16, AsymmetricNonSquarePooling2dInt16Test)


// Linear Activation
ARMNN_AUTO_TEST_CASE(ConstantLinearActivation, ConstantLinearActivationTest)
ARMNN_AUTO_TEST_CASE(ConstantLinearActivationUint8, ConstantLinearActivationUint8Test)
ARMNN_AUTO_TEST_CASE(ConstantLinearActivationInt16, ConstantLinearActivationInt16Test)

// Normalization
ARMNN_AUTO_TEST_CASE(SimpleNormalizationAcross, SimpleNormalizationAcrossTest)
ARMNN_AUTO_TEST_CASE(SimpleNormalizationWithin, SimpleNormalizationWithinTest)
ARMNN_AUTO_TEST_CASE(SimpleNormalizationAcrossNhwc, SimpleNormalizationAcrossNhwcTest)

// Softmax
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta1, SimpleSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta2, SimpleSoftmaxTest, 2.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta1Uint8, SimpleSoftmaxUint8Test, 1.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta2Uint8, SimpleSoftmaxUint8Test, 2.0f)

ARMNN_AUTO_TEST_CASE(Simple3dSoftmax, Simple3dSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE(Simple3dSoftmaxUint8, Simple3dSoftmaxUint8Test, 1.0f)

ARMNN_AUTO_TEST_CASE(Simple4dSoftmax, Simple4dSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE(Simple4dSoftmaxUint8, Simple4dSoftmaxUint8Test, 1.0f)

ARMNN_AUTO_TEST_CASE(SimpleSoftmaxUint16, SimpleSoftmaxUint16Test, 1.0f)
ARMNN_AUTO_TEST_CASE(Simple3dSoftmaxUint16, Simple3dSoftmaxUint16Test, 1.0f)
ARMNN_AUTO_TEST_CASE(Simple4dSoftmaxUint16, Simple4dSoftmaxUint16Test, 1.0f)

// Sigmoid Activation
ARMNN_AUTO_TEST_CASE(SimpleSigmoid, SimpleSigmoidTest)
ARMNN_AUTO_TEST_CASE(SimpleSigmoidUint8, SimpleSigmoidUint8Test)
ARMNN_AUTO_TEST_CASE(SimpleSigmoidInt16, SimpleSigmoidInt16Test)

// BoundedReLU Activation
ARMNN_AUTO_TEST_CASE(ReLu1, BoundedReLuUpperAndLowerBoundTest)
ARMNN_AUTO_TEST_CASE(ReLu6, BoundedReLuUpperBoundOnlyTest)
ARMNN_AUTO_TEST_CASE(ReLu1Uint8, BoundedReLuUint8UpperAndLowerBoundTest)
ARMNN_AUTO_TEST_CASE(ReLu6Uint8, BoundedReLuUint8UpperBoundOnlyTest)
ARMNN_AUTO_TEST_CASE(BoundedReLuInt16, BoundedReLuInt16Test)

// ReLU Activation
ARMNN_AUTO_TEST_CASE(ReLu, ReLuTest)
ARMNN_AUTO_TEST_CASE(ReLuUint8, ReLuUint8Test)
ARMNN_AUTO_TEST_CASE(ReLuInt16, ReLuInt16Test)

// SoftReLU Activation
ARMNN_AUTO_TEST_CASE(SoftReLu, SoftReLuTest)
ARMNN_AUTO_TEST_CASE(SoftReLuUint8, SoftReLuUint8Test)
ARMNN_AUTO_TEST_CASE(SoftReLuInt16, SoftReLuInt16Test)


// LeakyReLU Activation
ARMNN_AUTO_TEST_CASE(LeakyReLu, LeakyReLuTest)
ARMNN_AUTO_TEST_CASE(LeakyReLuUint8, LeakyReLuUint8Test)
ARMNN_AUTO_TEST_CASE(LeakyReLuInt16, LeakyReLuInt16Test)

// Abs Activation
ARMNN_AUTO_TEST_CASE(Abs, AbsTest)
ARMNN_AUTO_TEST_CASE(AbsUint8, AbsUint8Test)
ARMNN_AUTO_TEST_CASE(AbsInt16, AbsInt16Test)

// Sqrt Activation
ARMNN_AUTO_TEST_CASE(Sqrt, SqrtTest)
ARMNN_AUTO_TEST_CASE(SqrtUint8, SqrtUint8Test)
ARMNN_AUTO_TEST_CASE(SqrtInt16, SqrtInt16Test)

// Square Activation
ARMNN_AUTO_TEST_CASE(Square, SquareTest)
ARMNN_AUTO_TEST_CASE(SquareUint8, SquareUint8Test)
ARMNN_AUTO_TEST_CASE(SquareInt16, SquareInt16Test)

// Tanh Activation
ARMNN_AUTO_TEST_CASE(Tanh, TanhTest)
ARMNN_AUTO_TEST_CASE(TanhUint8, TanhUint8Test)
ARMNN_AUTO_TEST_CASE(TanhInt16, TanhInt16Test)


// Fully Connected
ARMNN_AUTO_TEST_CASE(SimpleFullyConnected, FullyConnectedFloat32Test, false, false)
ARMNN_AUTO_TEST_CASE(FullyConnectedUint8, FullyConnectedTest<armnn::DataType::QuantisedAsymm8>, false)
ARMNN_AUTO_TEST_CASE(FullyConnectedQSymm16, FullyConnectedTest<armnn::DataType::QuantisedSymm16>, false)
ARMNN_AUTO_TEST_CASE(SimpleFullyConnectedWithBias, FullyConnectedFloat32Test, true, false)
ARMNN_AUTO_TEST_CASE(FullyConnectedBiasedUint8, FullyConnectedTest<armnn::DataType::QuantisedAsymm8>, true)
ARMNN_AUTO_TEST_CASE(FullyConnectedBiasedQSymm16, FullyConnectedTest<armnn::DataType::QuantisedSymm16>, true)
ARMNN_AUTO_TEST_CASE(SimpleFullyConnectedWithTranspose, FullyConnectedFloat32Test, false, true)

ARMNN_AUTO_TEST_CASE(FullyConnectedLarge, FullyConnectedLargeTest, false)
ARMNN_AUTO_TEST_CASE(FullyConnectedLargeTransposed, FullyConnectedLargeTest, true)

// Splitter
ARMNN_AUTO_TEST_CASE(SimpleSplitter, SplitterTest)
ARMNN_AUTO_TEST_CASE(SimpleSplitterUint8, SplitterUint8Test)
ARMNN_AUTO_TEST_CASE(SimpleSplitterInt16, SplitterInt16Test)

ARMNN_AUTO_TEST_CASE(CopyViaSplitter, CopyViaSplitterTest)
ARMNN_AUTO_TEST_CASE(CopyViaSplitterUint8, CopyViaSplitterUint8Test)
ARMNN_AUTO_TEST_CASE(CopyViaSplitterInt16, CopyViaSplitterInt16Test)

// Concat
ARMNN_AUTO_TEST_CASE(SimpleConcat, ConcatTest)
ARMNN_AUTO_TEST_CASE(ConcatUint8, ConcatUint8Test)
ARMNN_AUTO_TEST_CASE(ConcatUint8DifferentQParams, ConcatUint8DifferentQParamsTest)
ARMNN_AUTO_TEST_CASE(ConcatUint16, ConcatUint16Test)

// Add
ARMNN_AUTO_TEST_CASE(SimpleAdd, AdditionTest)
ARMNN_AUTO_TEST_CASE(AddBroadcast1Element, AdditionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(AddBroadcast, AdditionBroadcastTest)

ARMNN_AUTO_TEST_CASE(AdditionUint8, AdditionUint8Test)
ARMNN_AUTO_TEST_CASE(AddBroadcastUint8, AdditionBroadcastUint8Test)
ARMNN_AUTO_TEST_CASE(AddBroadcast1ElementUint8, AdditionBroadcast1ElementUint8Test)

ARMNN_AUTO_TEST_CASE(AdditionInt16, AdditionInt16Test)
ARMNN_AUTO_TEST_CASE(AddBroadcastInt16, AdditionBroadcastInt16Test)
ARMNN_AUTO_TEST_CASE(AddBroadcast1ElementInt16, AdditionBroadcast1ElementInt16Test)

// Sub
ARMNN_AUTO_TEST_CASE(SimpleSub, SubtractionTest)
ARMNN_AUTO_TEST_CASE(SubBroadcast1Element, SubtractionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(SubBroadcast, SubtractionBroadcastTest)

ARMNN_AUTO_TEST_CASE(SubtractionUint8, SubtractionUint8Test)
ARMNN_AUTO_TEST_CASE(SubBroadcastUint8, SubtractionBroadcastUint8Test)
ARMNN_AUTO_TEST_CASE(SubBroadcast1ElementUint8, SubtractionBroadcast1ElementUint8Test)

ARMNN_AUTO_TEST_CASE(SubtractionInt16, SubtractionInt16Test)
ARMNN_AUTO_TEST_CASE(SubBroadcastInt16, SubtractionBroadcastInt16Test)
ARMNN_AUTO_TEST_CASE(SubBroadcast1ElementInt16, SubtractionBroadcast1ElementInt16Test)

// Div
ARMNN_AUTO_TEST_CASE(SimpleDivision, DivisionTest)
ARMNN_AUTO_TEST_CASE(DivisionByZero, DivisionByZeroTest)
ARMNN_AUTO_TEST_CASE(DivisionBroadcast1Element, DivisionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(DivisionBroadcast1DVector, DivisionBroadcast1DVectorTest)
// NOTE: division by zero for quantized div needs more attention
//       see IVGCVSW-1849
ARMNN_AUTO_TEST_CASE(DivisionUint8, DivisionUint8Test)
ARMNN_AUTO_TEST_CASE(DivisionUint8Broadcast1Element, DivisionBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(DivisionUint8Broadcast1DVector, DivisionBroadcast1DVectorUint8Test)

ARMNN_AUTO_TEST_CASE(DivisionInt16, DivisionInt16Test)
ARMNN_AUTO_TEST_CASE(DivisionInt16Broadcast1Element, DivisionBroadcast1ElementInt16Test)
ARMNN_AUTO_TEST_CASE(DivisionInt16Broadcast1DVector, DivisionBroadcast1DVectorInt16Test)

// Equal
ARMNN_AUTO_TEST_CASE(SimpleEqual, EqualSimpleTest)
ARMNN_AUTO_TEST_CASE(EqualBroadcast1Element, EqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(EqualBroadcast1DVector, EqualBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE(EqualUint8, EqualUint8Test)
ARMNN_AUTO_TEST_CASE(EqualBroadcast1ElementUint8, EqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(EqualBroadcast1DVectorUint8, EqualBroadcast1DVectorUint8Test)

// Greater
ARMNN_AUTO_TEST_CASE(SimpleGreater, GreaterSimpleTest)
ARMNN_AUTO_TEST_CASE(GreaterBroadcast1Element, GreaterBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(GreaterBroadcast1DVector, GreaterBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE(GreaterUint8, GreaterUint8Test)
ARMNN_AUTO_TEST_CASE(GreaterBroadcast1ElementUint8, GreaterBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(GreaterBroadcast1DVectorUint8, GreaterBroadcast1DVectorUint8Test)

// Max
ARMNN_AUTO_TEST_CASE(SimpleMaximum, MaximumSimpleTest)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1Element, MaximumBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVector, MaximumBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE(MaximumUint8, MaximumUint8Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1ElementUint8, MaximumBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVectorUint8, MaximumBroadcast1DVectorUint8Test)
ARMNN_AUTO_TEST_CASE(MaximumInt16, MaximumInt16Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1ElementInt16, MaximumBroadcast1ElementInt16Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVectorInt16, MaximumBroadcast1DVectorInt16Test)

// Min
ARMNN_AUTO_TEST_CASE(SimpleMinimum1, MinimumBroadcast1ElementTest1)
ARMNN_AUTO_TEST_CASE(SimpleMinimum2, MinimumBroadcast1ElementTest2)
ARMNN_AUTO_TEST_CASE(Minimum1DVectorUint8, MinimumBroadcast1DVectorUint8Test)
ARMNN_AUTO_TEST_CASE(MinimumInt16, MinimumInt16Test)
ARMNN_AUTO_TEST_CASE(MinimumBroadcast1ElementInt16, MinimumBroadcast1ElementInt16Test)
ARMNN_AUTO_TEST_CASE(MinimumBroadcast1DVectorInt16, MinimumBroadcast1DVectorInt16Test)

// Mul
ARMNN_AUTO_TEST_CASE(SimpleMultiplication, MultiplicationTest)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1Element, MultiplicationBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1DVector, MultiplicationBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE(MultiplicationUint8, MultiplicationUint8Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1ElementUint8, MultiplicationBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1DVectorUint8, MultiplicationBroadcast1DVectorUint8Test)
ARMNN_AUTO_TEST_CASE(MultiplicationInt16, MultiplicationInt16Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1ElementInt16, MultiplicationBroadcast1ElementInt16Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1DVectorInt16, MultiplicationBroadcast1DVectorInt16Test)

// Batch Norm
ARMNN_AUTO_TEST_CASE(BatchNorm, BatchNormTest)
ARMNN_AUTO_TEST_CASE(BatchNormNhwc, BatchNormNhwcTest)
ARMNN_AUTO_TEST_CASE(BatchNormUint8, BatchNormUint8Test)
ARMNN_AUTO_TEST_CASE(BatchNormUint8Nhwc, BatchNormUint8NhwcTest)
ARMNN_AUTO_TEST_CASE(BatchNormInt16, BatchNormInt16Test)
ARMNN_AUTO_TEST_CASE(BatchNormInt16Nhwc, BatchNormInt16NhwcTest)

// Resize Bilinear - NCHW
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinear,
                     SimpleResizeBilinearTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearUint8,
                     SimpleResizeBilinearTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNop,
                     ResizeBilinearNopTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopUint8,
                     ResizeBilinearNopTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMin,
                     ResizeBilinearSqMinTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinUint8,
                     ResizeBilinearSqMinTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMin,
                     ResizeBilinearMinTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinUint8,
                     ResizeBilinearMinTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMag,
                     ResizeBilinearMagTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagUint8,
                     ResizeBilinearMagTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)

// Resize Bilinear - NHWC
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopNhwc,
                     ResizeBilinearNopTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopUint8Nhwc,
                     ResizeBilinearNopTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearNhwc,
                     SimpleResizeBilinearTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearUint8Nhwc,
                     SimpleResizeBilinearTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinNhwc,
                     ResizeBilinearSqMinTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinUint8Nhwc,
                     ResizeBilinearSqMinTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinNhwc,
                     ResizeBilinearMinTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinUint8Nhwc,
                     ResizeBilinearMinTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagNhwc,
                     ResizeBilinearMagTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagUint8Nhwc,
                     ResizeBilinearMagTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)

// Fake Quantization
ARMNN_AUTO_TEST_CASE(FakeQuantization, FakeQuantizationTest)

// L2 Normalization
ARMNN_AUTO_TEST_CASE(L2Normalization1d, L2Normalization1dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization2d, L2Normalization2dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization3d, L2Normalization3dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization4d, L2Normalization4dTest, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(L2Normalization1dInt16, L2Normalization1dInt16Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization2dInt16, L2Normalization2dInt16Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization3dInt16, L2Normalization3dInt16Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization4dInt16, L2Normalization4dInt16Test, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(L2Normalization1dUint8, L2Normalization1dUint8Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization2dUint8, L2Normalization2dUint8Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization3dUint8, L2Normalization3dUint8Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization4dUint8, L2Normalization4dUint8Test, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(L2Normalization1dNhwc, L2Normalization1dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization2dNhwc, L2Normalization2dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization3dNhwc, L2Normalization3dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization4dNhwc, L2Normalization4dTest, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(L2Normalization1dInt16Nhwc, L2Normalization1dInt16Test, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization2dInt16Nhwc, L2Normalization2dInt16Test, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization3dInt16Nhwc, L2Normalization3dInt16Test, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization4dInt16Nhwc, L2Normalization4dInt16Test, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(L2Normalization1dUint8Nhwc, L2Normalization1dUint8Test, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization2dUint8Nhwc, L2Normalization2dUint8Test, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization3dUint8Nhwc, L2Normalization3dUint8Test, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization4dUint8Nhwc, L2Normalization4dUint8Test, armnn::DataLayout::NHWC)

// Pad
ARMNN_AUTO_TEST_CASE(PadFloat322d, PadFloat322dTest)
ARMNN_AUTO_TEST_CASE(PadFloat323d, PadFloat323dTest)
ARMNN_AUTO_TEST_CASE(PadFloat324d, PadFloat324dTest)

ARMNN_AUTO_TEST_CASE(PadUint82d, PadUint82dTest)
ARMNN_AUTO_TEST_CASE(PadUint83d, PadUint83dTest)
ARMNN_AUTO_TEST_CASE(PadUint84d, PadUint84dTest)

// Constant
ARMNN_AUTO_TEST_CASE(Constant, ConstantTest)
ARMNN_AUTO_TEST_CASE(ConstantUint8, ConstantUint8CustomQuantizationScaleAndOffsetTest)
ARMNN_AUTO_TEST_CASE(ConstantInt16, ConstantInt16CustomQuantizationScaleAndOffsetTest)

// Concat
ARMNN_AUTO_TEST_CASE(Concatenation1d, Concatenation1dTest)
ARMNN_AUTO_TEST_CASE(Concatenation1dUint8, Concatenation1dUint8Test)

ARMNN_AUTO_TEST_CASE(Concatenation2dDim0, Concatenation2dDim0Test)
ARMNN_AUTO_TEST_CASE(Concatenation2dDim0Uint8, Concatenation2dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation2dDim1, Concatenation2dDim1Test)
ARMNN_AUTO_TEST_CASE(Concatenation2dDim1Uint8, Concatenation2dDim1Uint8Test)

ARMNN_AUTO_TEST_CASE(Concatenation2dDim0DiffInputDims, Concatenation2dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation2dDim0DiffInputDimsUint8, Concatenation2dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation2dDim1DiffInputDims, Concatenation2dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation2dDim1DiffInputDimsUint8, Concatenation2dDim1DiffInputDimsUint8Test)

ARMNN_AUTO_TEST_CASE(Concatenation3dDim0, Concatenation3dDim0Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim0Uint8, Concatenation3dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1, Concatenation3dDim1Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1Uint8, Concatenation3dDim1Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2, Concatenation3dDim2Test, true)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2Uint8, Concatenation3dDim2Uint8Test, true)

ARMNN_AUTO_TEST_CASE(Concatenation3dDim0DiffInputDims, Concatenation3dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim0DiffInputDimsUint8, Concatenation3dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1DiffInputDims, Concatenation3dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1DiffInputDimsUint8, Concatenation3dDim1DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2DiffInputDims, Concatenation3dDim2DiffInputDimsTest, true)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2DiffInputDimsUint8, Concatenation3dDim2DiffInputDimsUint8Test, true)

ARMNN_AUTO_TEST_CASE(Concatenation4dDim0, Concatenation4dDim0Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim1, Concatenation4dDim1Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim2, Concatenation4dDim2Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim3, Concatenation4dDim3Test, true)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim0Uint8, Concatenation4dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim1Uint8, Concatenation4dDim1Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim2Uint8, Concatenation4dDim2Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim3Uint8, Concatenation4dDim3Uint8Test, true)

ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim0, Concatenation4dDiffShapeDim0Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim1, Concatenation4dDiffShapeDim1Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim2, Concatenation4dDiffShapeDim2Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim3, Concatenation4dDiffShapeDim3Test, true)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim0Uint8, Concatenation4dDiffShapeDim0Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim1Uint8, Concatenation4dDiffShapeDim1Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim2Uint8, Concatenation4dDiffShapeDim2Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim3Uint8, Concatenation4dDiffShapeDim3Uint8Test, true)

// Floor
ARMNN_AUTO_TEST_CASE(SimpleFloor, SimpleFloorTest<armnn::DataType::Float32>)
ARMNN_AUTO_TEST_CASE(SimpleFloorQuantisedSymm16, SimpleFloorTest<armnn::DataType::QuantisedSymm16>)

// Reshape
ARMNN_AUTO_TEST_CASE(SimpleReshapeFloat32, SimpleReshapeTest<armnn::DataType::Float32>)
ARMNN_AUTO_TEST_CASE(SimpleReshapeQuantisedAsymm8, SimpleReshapeTest<armnn::DataType::QuantisedAsymm8>)
ARMNN_AUTO_TEST_CASE(SimpleReshapeQuantisedSymm16, SimpleReshapeTest<armnn::DataType::QuantisedSymm16>)

// Rsqrt
ARMNN_AUTO_TEST_CASE(Rsqrt2d, Rsqrt2dTest<armnn::DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Rsqrt3d, Rsqrt3dTest<armnn::DataType::Float32>)
ARMNN_AUTO_TEST_CASE(RsqrtZero, RsqrtZeroTest<armnn::DataType::Float32>)
ARMNN_AUTO_TEST_CASE(RsqrtNegative, RsqrtNegativeTest<armnn::DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Rsqrt2dQuantisedAsymm8, Rsqrt2dTest<armnn::DataType::QuantisedAsymm8>)
ARMNN_AUTO_TEST_CASE(Rsqrt3dQuantisedAsymm8, Rsqrt3dTest<armnn::DataType::QuantisedAsymm8>)
ARMNN_AUTO_TEST_CASE(Rsqrt2dQuantisedSymm16, Rsqrt2dTest<armnn::DataType::QuantisedSymm16>)
ARMNN_AUTO_TEST_CASE(Rsqrt3dQuantisedSymm16, Rsqrt3dTest<armnn::DataType::QuantisedSymm16>)

// Permute
ARMNN_AUTO_TEST_CASE(SimplePermuteFloat32, SimplePermuteFloat32Test)
ARMNN_AUTO_TEST_CASE(SimplePermuteUint8, SimplePermuteUint8Test)
ARMNN_AUTO_TEST_CASE(PermuteFloat32ValueSet1, PermuteFloat32ValueSet1Test)
ARMNN_AUTO_TEST_CASE(PermuteFloat32ValueSet2, PermuteFloat32ValueSet2Test)
ARMNN_AUTO_TEST_CASE(PermuteFloat32ValueSet3, PermuteFloat32ValueSet3Test)

// Lstm
ARMNN_AUTO_TEST_CASE(LstmLayerFloat32WithCifgWithPeepholeNoProjection,
                     LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerFloat32NoCifgNoPeepholeNoProjection,
                     LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerFloat32NoCifgWithPeepholeWithProjection,
                     LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest)

ARMNN_AUTO_TEST_CASE(LstmLayerInt16NoCifgNoPeepholeNoProjection,
                     LstmLayerInt16NoCifgNoPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerInt16WithCifgWithPeepholeNoProjection,
                     LstmLayerInt16WithCifgWithPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerInt16NoCifgWithPeepholeWithProjection,
                     LstmLayerInt16NoCifgWithPeepholeWithProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerInt16NoCifgNoPeepholeNoProjectionInt16Constant,
                     LstmLayerInt16NoCifgNoPeepholeNoProjectionInt16ConstantTest)

// Convert from Float16 to Float32
ARMNN_AUTO_TEST_CASE(SimpleConvertFp16ToFp32, SimpleConvertFp16ToFp32Test)
// Convert from Float32 to Float16
ARMNN_AUTO_TEST_CASE(SimpleConvertFp32ToFp16, SimpleConvertFp32ToFp16Test)

// Mean
ARMNN_AUTO_TEST_CASE(MeanUint8Simple, MeanUint8SimpleTest)
ARMNN_AUTO_TEST_CASE(MeanUint8SimpleAxis, MeanUint8SimpleAxisTest)
ARMNN_AUTO_TEST_CASE(MeanUint8KeepDims, MeanUint8KeepDimsTest)
ARMNN_AUTO_TEST_CASE(MeanUint8MultipleDims, MeanUint8MultipleDimsTest)
ARMNN_AUTO_TEST_CASE(MeanVtsUint8, MeanVtsUint8Test)

ARMNN_AUTO_TEST_CASE(MeanFloatSimple, MeanFloatSimpleTest)
ARMNN_AUTO_TEST_CASE(MeanFloatSimpleAxis, MeanFloatSimpleAxisTest)
ARMNN_AUTO_TEST_CASE(MeanFloatKeepDims, MeanFloatKeepDimsTest)
ARMNN_AUTO_TEST_CASE(MeanFloatMultipleDims, MeanFloatMultipleDimsTest)
ARMNN_AUTO_TEST_CASE(MeanVtsFloat1, MeanVtsFloat1Test)
ARMNN_AUTO_TEST_CASE(MeanVtsFloat2, MeanVtsFloat2Test)
ARMNN_AUTO_TEST_CASE(MeanVtsFloat3, MeanVtsFloat3Test)

ARMNN_AUTO_TEST_CASE(AdditionAfterMaxPool, AdditionAfterMaxPoolTest)

// Space To Batch Nd
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdSimpleFloat32, SpaceToBatchNdSimpleFloat32Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiChannelsFloat32, SpaceToBatchNdMultiChannelsFloat32Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiBlockFloat32, SpaceToBatchNdMultiBlockFloat32Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdPaddingFloat32, SpaceToBatchNdPaddingFloat32Test)

ARMNN_AUTO_TEST_CASE(SpaceToBatchNdSimpleUint8, SpaceToBatchNdSimpleUint8Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiChannelsUint8, SpaceToBatchNdMultiChannelsUint8Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiBlockUint8, SpaceToBatchNdMultiBlockUint8Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdPaddingUint8, SpaceToBatchNdPaddingUint8Test)

ARMNN_AUTO_TEST_CASE(SpaceToBatchNdSimpleNHWCFloat32, SpaceToBatchNdSimpleNHWCFloat32Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiChannelsNHWCFloat32, SpaceToBatchNdMultiChannelsNHWCFloat32Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiBlockNHWCFloat32, SpaceToBatchNdMultiBlockNHWCFloat32Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdPaddingNHWCFloat32, SpaceToBatchNdPaddingNHWCFloat32Test)

ARMNN_AUTO_TEST_CASE(SpaceToBatchNdSimpleNHWCUint8, SpaceToBatchNdSimpleNHWCUint8Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiChannelsNHWCUint8, SpaceToBatchNdMultiChannelsNHWCUint8Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiBlockNHWCUint8, SpaceToBatchNdMultiBlockNHWCUint8Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdPaddingNHWCUint8, SpaceToBatchNdPaddingNHWCUint8Test)

ARMNN_AUTO_TEST_CASE(SpaceToBatchNdSimpleUint16, SpaceToBatchNdSimpleUint16Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiChannelsUint16, SpaceToBatchNdMultiChannelsUint16Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiBlockUint16, SpaceToBatchNdMultiBlockUint16Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdPaddingUint16, SpaceToBatchNdPaddingUint16Test)

ARMNN_AUTO_TEST_CASE(SpaceToBatchNdSimpleNHWCUint16, SpaceToBatchNdSimpleNHWCUint16Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiChannelsNHWCUint16, SpaceToBatchNdMultiChannelsNHWCUint16Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdMultiBlockNHWCUint16, SpaceToBatchNdMultiBlockNHWCUint16Test)
ARMNN_AUTO_TEST_CASE(SpaceToBatchNdPaddingNHWCUint16, SpaceToBatchNdPaddingNHWCUint16Test)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat321, BatchToSpaceNdNhwcFloat32Test1)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat322, BatchToSpaceNdNhwcFloat32Test2)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat323, BatchToSpaceNdNhwcFloat32Test3)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat324, BatchToSpaceNdNhwcFloat32Test4)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat321, BatchToSpaceNdNchwFloat32Test1)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat322, BatchToSpaceNdNchwFloat32Test2)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat323, BatchToSpaceNdNchwFloat32Test3)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint1, BatchToSpaceNdNhwcUintTest1)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint2, BatchToSpaceNdNhwcUintTest2)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint3, BatchToSpaceNdNhwcUintTest3)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint1, BatchToSpaceNdNchwUintTest1)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint2, BatchToSpaceNdNchwUintTest2)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint3, BatchToSpaceNdNchwUintTest3)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint4, BatchToSpaceNdNchwUintTest4)

// Strided Slice
ARMNN_AUTO_TEST_CASE(StridedSlice4DFloat32, StridedSlice4DFloat32Test)
ARMNN_AUTO_TEST_CASE(StridedSlice4DReverseFloat32, StridedSlice4DReverseFloat32Test)
ARMNN_AUTO_TEST_CASE(StridedSliceSimpleStrideFloat32, StridedSliceSimpleStrideFloat32Test)
ARMNN_AUTO_TEST_CASE(StridedSliceSimpleRangeMaskFloat32, StridedSliceSimpleRangeMaskFloat32Test)
ARMNN_AUTO_TEST_CASE(StridedSliceShrinkAxisMaskFloat32, StridedSliceShrinkAxisMaskFloat32Test)
ARMNN_AUTO_TEST_CASE(StridedSlice3DFloat32, StridedSlice3DFloat32Test)
ARMNN_AUTO_TEST_CASE(StridedSlice3DReverseFloat32, StridedSlice3DReverseFloat32Test)
ARMNN_AUTO_TEST_CASE(StridedSlice2DFloat32, StridedSlice2DFloat32Test)
ARMNN_AUTO_TEST_CASE(StridedSlice2DReverseFloat32, StridedSlice2DReverseFloat32Test)

ARMNN_AUTO_TEST_CASE(StridedSlice4DUint8, StridedSlice4DUint8Test)
ARMNN_AUTO_TEST_CASE(StridedSlice4DReverseUint8, StridedSlice4DReverseUint8Test)
ARMNN_AUTO_TEST_CASE(StridedSliceSimpleStrideUint8, StridedSliceSimpleStrideUint8Test)
ARMNN_AUTO_TEST_CASE(StridedSliceSimpleRangeMaskUint8, StridedSliceSimpleRangeMaskUint8Test)
ARMNN_AUTO_TEST_CASE(StridedSliceShrinkAxisMaskUint8, StridedSliceShrinkAxisMaskUint8Test)
ARMNN_AUTO_TEST_CASE(StridedSlice3DUint8, StridedSlice3DUint8Test)
ARMNN_AUTO_TEST_CASE(StridedSlice3DReverseUint8, StridedSlice3DReverseUint8Test)
ARMNN_AUTO_TEST_CASE(StridedSlice2DUint8, StridedSlice2DUint8Test)
ARMNN_AUTO_TEST_CASE(StridedSlice2DReverseUint8, StridedSlice2DReverseUint8Test)

ARMNN_AUTO_TEST_CASE(StridedSlice4DInt16, StridedSlice4DInt16Test)
ARMNN_AUTO_TEST_CASE(StridedSlice4DReverseInt16, StridedSlice4DReverseInt16Test)
ARMNN_AUTO_TEST_CASE(StridedSliceSimpleStrideInt16, StridedSliceSimpleStrideInt16Test)
ARMNN_AUTO_TEST_CASE(StridedSliceSimpleRangeMaskInt16, StridedSliceSimpleRangeMaskInt16Test)
ARMNN_AUTO_TEST_CASE(StridedSliceShrinkAxisMaskInt16, StridedSliceShrinkAxisMaskInt16Test)
ARMNN_AUTO_TEST_CASE(StridedSlice3DInt16, StridedSlice3DInt16Test)
ARMNN_AUTO_TEST_CASE(StridedSlice3DReverseInt16, StridedSlice3DReverseInt16Test)
ARMNN_AUTO_TEST_CASE(StridedSlice2DInt16, StridedSlice2DInt16Test)
ARMNN_AUTO_TEST_CASE(StridedSlice2DReverseInt16, StridedSlice2DReverseInt16Test)

// Debug
ARMNN_AUTO_TEST_CASE(Debug4DFloat32, Debug4DFloat32Test)
ARMNN_AUTO_TEST_CASE(Debug3DFloat32, Debug3DFloat32Test)
ARMNN_AUTO_TEST_CASE(Debug2DFloat32, Debug2DFloat32Test)
ARMNN_AUTO_TEST_CASE(Debug1DFloat32, Debug1DFloat32Test)

ARMNN_AUTO_TEST_CASE(Debug4DUint8, Debug4DUint8Test)
ARMNN_AUTO_TEST_CASE(Debug3DUint8, Debug3DUint8Test)
ARMNN_AUTO_TEST_CASE(Debug2DUint8, Debug2DUint8Test)
ARMNN_AUTO_TEST_CASE(Debug1DUint8, Debug1DUint8Test)

// Gather
ARMNN_AUTO_TEST_CASE(Gather1DParamsFloat, Gather1DParamsFloatTest)
ARMNN_AUTO_TEST_CASE(Gather1DParamsUint8, Gather1DParamsUint8Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsFloat, GatherMultiDimParamsFloatTest)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsUint8, GatherMultiDimParamsUint8Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsMultiDimIndicesFloat, GatherMultiDimParamsMultiDimIndicesFloatTest)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsMultiDimIndicesUint8, GatherMultiDimParamsMultiDimIndicesUint8Test)

// Detection PostProcess
BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsFloat)
{
    DetectionPostProcessRegularNmsFloatTest<armnn::RefWorkloadFactory>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsFloat)
{
    DetectionPostProcessFastNmsFloatTest<armnn::RefWorkloadFactory>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsUint8)
{
    DetectionPostProcessRegularNmsQuantizedTest<
        armnn::RefWorkloadFactory, armnn::DataType::QuantisedAsymm8>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsUint8)
{
    DetectionPostProcessRegularNmsQuantizedTest<
        armnn::RefWorkloadFactory, armnn::DataType::QuantisedAsymm8>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsInt16)
{
    DetectionPostProcessRegularNmsQuantizedTest<
        armnn::RefWorkloadFactory, armnn::DataType::QuantisedSymm16>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsInt16)
{
    DetectionPostProcessFastNmsQuantizedTest<
        armnn::RefWorkloadFactory, armnn::DataType::QuantisedSymm16>();
}

// Dequantize
ARMNN_AUTO_TEST_CASE(DequantizeSimpleUint8, DequantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE(DequantizeOffsetUint8, DequantizeOffsetUint8Test)
ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt16, DequantizeSimpleInt16Test)

// Quantize
ARMNN_AUTO_TEST_CASE(QuantizeSimpleUint8, QuantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE(QuantizeClampUint8, QuantizeClampUint8Test)
ARMNN_AUTO_TEST_CASE(QuantizeClampInt16, QuantizeClampInt16Test)

BOOST_AUTO_TEST_SUITE_END()
