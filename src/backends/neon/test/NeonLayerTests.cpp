//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <test/TensorHelpers.hpp>
#include <test/UnitTests.hpp>

#include <backendsCommon/CpuTensorHandle.hpp>
#include <neon/NeonLayerSupport.hpp>
#include <neon/NeonWorkloadFactory.hpp>
#include <reference/RefWorkloadFactory.hpp>
#include <backendsCommon/test/ActivationFixture.hpp>
#include <backendsCommon/test/LayerTests.hpp>
#include <backendsCommon/test/TensorCopyUtils.hpp>
#include <backendsCommon/test/WorkloadTestUtils.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Compute_ArmComputeNeon)
using FactoryType = armnn::NeonWorkloadFactory;

// ============================================================================
// UNIT tests

// Convolution
ARMNN_AUTO_TEST_CASE(SimpleConvolution1d, Convolution1dTest, true)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2d, SimpleConvolution2d3x5Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2dNhwc, SimpleConvolution2d3x5Test, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x3Uint8, SimpleConvolution2d3x3Uint8Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x3Uint8Nhwc, SimpleConvolution2d3x3Uint8Test, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2d, SimpleConvolution2d3x5Test, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dNhwc, SimpleConvolution2d3x5Test, false, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dStride2x2Nhwc,
                     SimpleConvolution2d3x3Stride2x2Test, false, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dSquare, SimpleConvolution2d3x3Test, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2dAsymmetricPadding, Convolution2dAsymmetricPaddingTest, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dSquareNhwc, SimpleConvolution2d3x3Test, false, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2dAsymmetricPaddingNhwc,
                     Convolution2dAsymmetricPaddingTest,
                     armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2dSquareNhwc, SimpleConvolution2d3x3NhwcTest, false)

// Depthwise Convolution
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test, false, armnn::DataLayout::NCHW)

// NHWC Depthwise Convolution
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1NHhwc,
                     DepthwiseConvolution2dDepthMul1Test, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1Nhwc,
                     DepthwiseConvolution2dDepthMul1Test, false, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test, false, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthNhwc, DepthwiseConvolution2dDepthNhwcTest, false)
ARMNN_AUTO_TEST_CASE(SimpleDepthwiseConvolution2d3x3Dilation3x3Nhwc,
                     SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest)


ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dAsymmetric,
                     DepthwiseConvolution2dAsymmetricTest, true, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dAsymmetric,
                     DepthwiseConvolution2dAsymmetricTest, false, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dAsymmetricNhwc,
                     DepthwiseConvolution2dAsymmetricTest, true, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dAsymmetricNhwc,
                     DepthwiseConvolution2dAsymmetricTest, false, armnn::DataLayout::NHWC)

namespace
{

armnn::DepthwiseConvolution2dDescriptor MakeDepthwiseConv2dDesc(uint32_t strideX, uint32_t strideY,
    uint32_t depthMultiplier = 1, uint32_t padLeft = 0, uint32_t padRight = 0,
    uint32_t padTop = 0, uint32_t padBottom = 0)
{
    boost::ignore_unused(depthMultiplier);

    armnn::DepthwiseConvolution2dDescriptor desc;

    desc.m_PadLeft = padLeft;
    desc.m_PadRight = padRight;

    desc.m_PadTop = padTop;
    desc.m_PadBottom = padBottom;
    desc.m_StrideX = strideX;
    desc.m_StrideY = strideY;
    desc.m_BiasEnabled = false;

    return desc;
}

armnn::TensorInfo CreateOutputTensorInfo(const armnn::TensorInfo& inputInfo,
                                         const armnn::TensorInfo& weightsInfo,
                                         const armnn::DepthwiseConvolution2dDescriptor& descriptor,
                                         armnn::DataType dataType)
{
    const armnn::TensorShape& inputShape  = inputInfo.GetShape();
    const armnn::TensorShape& filterShape = weightsInfo.GetShape();

    unsigned int inWidth = inputShape[3];
    unsigned int inHeight = inputShape[2];
    unsigned int inBatchSize = inputShape[0];

    unsigned int filterWidth = filterShape[3];
    unsigned int readWidth = (inWidth + descriptor.m_PadLeft + descriptor.m_PadRight) - (filterWidth);
    unsigned int outWidth =  1u + (readWidth / descriptor.m_StrideX);

    unsigned int filterHeight = filterShape[2];
    unsigned int readHeight = (inHeight + descriptor.m_PadTop + descriptor.m_PadBottom) - (filterHeight);
    unsigned int outHeight = 1u + (readHeight / descriptor.m_StrideY);
    unsigned int depthMultiplier = filterShape[0];

    unsigned int outChannels = filterShape[1] * depthMultiplier;
    unsigned int outBatchSize = inBatchSize;

    armnn::TensorShape outputShape({outBatchSize, outChannels, outHeight, outWidth});
    return armnn::TensorInfo(outputShape, dataType);
}
}

BOOST_AUTO_TEST_CASE(DepthwiseConv2dUtils)
{
    const armnn::DataType dataType = armnn::DataType::Float32;

    armnn::TensorInfo inputInfo({1, 1, 10, 10 }, dataType);
    armnn::TensorInfo outputInfo;
    armnn::TensorInfo weightsInfo3x3({ 1, 1, 3, 3 }, dataType);
    armnn::TensorInfo biasesInfo;

    armnn::DepthwiseConvolution2dDescriptor descriptor;
    armnn::NeonLayerSupport layerSupport;

    // Strides supported: 1,2,3
    descriptor = MakeDepthwiseConv2dDesc(1, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(1, 2);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(1, 3);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(2, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(2, 2);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(2, 3);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(3, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(3, 2);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(3, 3);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    // Supported stride 4
    descriptor = MakeDepthwiseConv2dDesc(4, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    // Supported weights shape 1x1
    armnn::TensorInfo weightsInfo1x1({ 1, 1, 1, 1 }, armnn::DataType::Float32);
    descriptor = MakeDepthwiseConv2dDesc(1, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo1x1, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo1x1, biasesInfo));

    // Supported shape 2x2
    armnn::TensorInfo weightsInfo2x2({ 1, 1, 2, 2 }, armnn::DataType::Float32);
    descriptor = MakeDepthwiseConv2dDesc(1, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo2x2, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo2x2, biasesInfo));

    // Asymmetric padding
    descriptor = MakeDepthwiseConv2dDesc(1, 1, 1, 1, 2, 1, 2);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    BOOST_TEST(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));
}

// Dequantize
ARMNN_AUTO_TEST_CASE(DequantizeSimpleUint8, DequantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE(DequantizeOffsetUint8, DequantizeOffsetUint8Test)

// Pooling
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4, SimpleMaxPooling2dSize3x3Stride2x4Test, true)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4Uint8, SimpleMaxPooling2dSize3x3Stride2x4Uint8Test, true)

ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2d, SimpleMaxPooling2dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dNhwc, SimpleMaxPooling2dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dUint8, SimpleMaxPooling2dUint8Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dUint8Nhwc, SimpleMaxPooling2dUint8Test, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2d, SimpleAveragePooling2dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dNhwc, SimpleAveragePooling2dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dUint8, SimpleAveragePooling2dUint8Test, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dUint8Nhwc, SimpleAveragePooling2dUint8Test, armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(LargeTensorsAveragePooling2d, LargeTensorsAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE(LargeTensorsAveragePooling2dUint8, LargeTensorsAveragePooling2dUint8Test)

ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2d, SimpleL2Pooling2dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dNeon, SimpleL2Pooling2dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_SimpleL2Pooling2dUint8, SimpleL2Pooling2dUint8Test, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(L2Pooling2dSize3Stride1, L2Pooling2dSize3Stride1Test)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_L2Pooling2dSize3Stride1Uint8, L2Pooling2dSize3Stride1Uint8Test)
ARMNN_AUTO_TEST_CASE(L2Pooling2dSize3Stride3, L2Pooling2dSize3Stride3Test)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_L2Pooling2dSize3Stride3Uint8, L2Pooling2dSize3Stride3Uint8Test)
ARMNN_AUTO_TEST_CASE(L2Pooling2dSize3Stride4, L2Pooling2dSize3Stride4Test)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_L2Pooling2dSize3Stride4Uint8, L2Pooling2dSize3Stride4Uint8Test)
ARMNN_AUTO_TEST_CASE(L2Pooling2dSize7, L2Pooling2dSize7Test)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_L2Pooling2dSize7Uint8, L2Pooling2dSize7Uint8Test)
ARMNN_AUTO_TEST_CASE(L2Pooling2dSize9, L2Pooling2dSize9Test)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_L2Pooling2dSize9Uint8, L2Pooling2dSize9Uint8Test)

// Ignore padding values for pooling but count padding fields into the divisor
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleMaxPooling2d, IgnorePaddingSimpleMaxPooling2dTest)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleMaxPooling2dUint8, IgnorePaddingSimpleMaxPooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingMaxPooling2dSize3, IgnorePaddingMaxPooling2dSize3Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingMaxPooling2dSize3Uint8, IgnorePaddingMaxPooling2dSize3Uint8Test)

ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2d, IgnorePaddingSimpleAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2dUint8, IgnorePaddingSimpleAveragePooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2dNoPadding, IgnorePaddingSimpleAveragePooling2dNoPaddingTest)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleAveragePooling2dNoPaddingUint8,
    IgnorePaddingSimpleAveragePooling2dNoPaddingUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3, IgnorePaddingAveragePooling2dSize3Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3Uint8, IgnorePaddingAveragePooling2dSize3Uint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3x2Stride2x2,
                             IgnorePaddingAveragePooling2dSize3x2Stride2x2Test, false)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3x2Stride2x2NoPadding,
                             IgnorePaddingAveragePooling2dSize3x2Stride2x2Test,
                                          true)

ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleL2Pooling2d, IgnorePaddingSimpleL2Pooling2dTest)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_IgnorePaddingSimpleL2Pooling2dUint8, IgnorePaddingSimpleL2Pooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingL2Pooling2dSize3, IgnorePaddingL2Pooling2dSize3Test)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_IgnorePaddingL2Pooling2dSize3Uint8, IgnorePaddingL2Pooling2dSize3Uint8Test)

// Activation
ARMNN_AUTO_TEST_CASE(ConstantLinearActivation, ConstantLinearActivationTest)

// ReLu
ARMNN_AUTO_TEST_CASE(ReLu1Uint8, BoundedReLuUint8UpperAndLowerBoundTest)
ARMNN_AUTO_TEST_CASE(ReLu6Uint8, BoundedReLuUint8UpperBoundOnlyTest)

// Sigmoid
ARMNN_AUTO_TEST_CASE(SimpleSigmoid, SimpleSigmoidTest)
ARMNN_AUTO_TEST_CASE(SimpleSigmoidUint8, SimpleSigmoidUint8Test)

// Softmax
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta1, SimpleSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta2, SimpleSoftmaxTest, 2.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta1Uint8, SimpleSoftmaxUint8Test, 1.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta2Uint8, SimpleSoftmaxUint8Test, 2.0f)

ARMNN_AUTO_TEST_CASE(Simple3dSoftmaxBeta1, Simple3dSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE(Simple3dSoftmaxBeta1Uint8, Simple3dSoftmaxUint8Test, 1.0f)

ARMNN_AUTO_TEST_CASE(Simple4dSoftmaxBeta1, Simple4dSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE(Simple4dSoftmaxBeta1Uint8, Simple4dSoftmaxUint8Test, 1.0f)

// Splitter
ARMNN_AUTO_TEST_CASE(SimpleSplitter, SplitterTest)
ARMNN_AUTO_TEST_CASE(SimpleSplitterUint8, SplitterUint8Test)

ARMNN_AUTO_TEST_CASE(CopyViaSplitter, CopyViaSplitterTest)
ARMNN_AUTO_TEST_CASE(CopyViaSplitterUint8, CopyViaSplitterUint8Test)

// Concat
ARMNN_AUTO_TEST_CASE(SimpleConcat, ConcatTest)
ARMNN_AUTO_TEST_CASE(ConcatUint8, ConcatUint8Test)

// Fully Connected
ARMNN_AUTO_TEST_CASE(SimpleFullyConnected, FullyConnectedFloat32Test, false, false)
ARMNN_AUTO_TEST_CASE(SimpleFullyConnectedWithBias, FullyConnectedFloat32Test, true, false)
ARMNN_AUTO_TEST_CASE(SimpleFullyConnectedWithTranspose, FullyConnectedFloat32Test, false, true)
ARMNN_AUTO_TEST_CASE(FullyConnectedLarge, FullyConnectedLargeTest, false)
ARMNN_AUTO_TEST_CASE(FullyConnectedLargeTransposed, FullyConnectedLargeTest, true)
ARMNN_AUTO_TEST_CASE(FullyConnectedUint8, FullyConnectedTest<armnn::DataType::QuantisedAsymm8>, false)
ARMNN_AUTO_TEST_CASE(FullyConnectedBiasedUint8, FullyConnectedTest<armnn::DataType::QuantisedAsymm8>, true)

// Add
ARMNN_AUTO_TEST_CASE(SimpleAdd, AdditionTest)
ARMNN_AUTO_TEST_CASE(AddBroadcast, AdditionBroadcastTest)
ARMNN_AUTO_TEST_CASE(AddBroadcast1Element, AdditionBroadcast1ElementTest)

// Sub
ARMNN_AUTO_TEST_CASE(SimpleSub, SubtractionTest)
ARMNN_AUTO_TEST_CASE(SubBroadcast1Element, SubtractionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(SubBroadcast, SubtractionBroadcastTest)
ARMNN_AUTO_TEST_CASE(SubtractionUint8, SubtractionUint8Test)
ARMNN_AUTO_TEST_CASE(SubBroadcastUint8, SubtractionBroadcastUint8Test)
ARMNN_AUTO_TEST_CASE(SubBroadcast1ElementUint8, SubtractionBroadcast1ElementUint8Test)

// Mul
ARMNN_AUTO_TEST_CASE(SimpleMultiplication, MultiplicationTest)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1Element, MultiplicationBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1DVector, MultiplicationBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE(MultiplicationUint8, MultiplicationUint8Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1ElementUint8, MultiplicationBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1DVectorUint8, MultiplicationBroadcast1DVectorUint8Test)

// Batch Norm
ARMNN_AUTO_TEST_CASE(BatchNorm, BatchNormTest)
ARMNN_AUTO_TEST_CASE(BatchNormNhwc, BatchNormNhwcTest)

// Constant
ARMNN_AUTO_TEST_CASE(Constant, ConstantTest)
ARMNN_AUTO_TEST_CASE(ConstantUint8, ConstantUint8SimpleQuantizationScaleNoOffsetTest)

// Concatenation
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
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2, Concatenation3dDim2Test, false)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2Uint8, Concatenation3dDim2Uint8Test, false)

ARMNN_AUTO_TEST_CASE(Concatenation3dDim0DiffInputDims, Concatenation3dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim0DiffInputDimsUint8, Concatenation3dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1DiffInputDims, Concatenation3dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1DiffInputDimsUint8, Concatenation3dDim1DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2DiffInputDims, Concatenation3dDim2DiffInputDimsTest, false)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2DiffInputDimsUint8, Concatenation3dDim2DiffInputDimsUint8Test, false)

ARMNN_AUTO_TEST_CASE(Concatenation4dDim0, Concatenation4dDim0Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim1, Concatenation4dDim1Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim3, Concatenation4dDim3Test, false)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim0Uint8, Concatenation4dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim1Uint8, Concatenation4dDim1Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDim3Uint8, Concatenation4dDim3Uint8Test, false)

ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim0, Concatenation4dDiffShapeDim0Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim1, Concatenation4dDiffShapeDim1Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim3, Concatenation4dDiffShapeDim3Test, false)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim0Uint8, Concatenation4dDiffShapeDim0Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim1Uint8, Concatenation4dDiffShapeDim1Uint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation4dDiffShapeDim3Uint8, Concatenation4dDiffShapeDim3Uint8Test, false)
// L2 Normalization
ARMNN_AUTO_TEST_CASE(L2Normalization1d, L2Normalization1dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization2d, L2Normalization2dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization3d, L2Normalization3dTest, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization4d, L2Normalization4dTest, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(L2Normalization1dNhwc, L2Normalization1dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization2dNhwc, L2Normalization2dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization3dNhwc, L2Normalization3dTest, armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization4dNhwc, L2Normalization4dTest, armnn::DataLayout::NHWC)

// Floor
ARMNN_AUTO_TEST_CASE(SimpleFloor, SimpleFloorTest<armnn::DataType::Float32>)

// Greater
ARMNN_AUTO_TEST_CASE(SimpleGreater, GreaterSimpleTest)
ARMNN_AUTO_TEST_CASE(GreaterBroadcast1Element, GreaterBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(GreaterBroadcast1DVector, GreaterBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE(GreaterUint8, GreaterUint8Test)
ARMNN_AUTO_TEST_CASE(GreaterBroadcast1ElementUint8, GreaterBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(GreaterBroadcast1DVectorUint8, GreaterBroadcast1DVectorUint8Test)

// Reshape
ARMNN_AUTO_TEST_CASE(SimpleReshapeFloat32, SimpleReshapeTest<armnn::DataType::Float32>)
ARMNN_AUTO_TEST_CASE(SimpleReshapeUint8, SimpleReshapeTest<armnn::DataType::QuantisedAsymm8>)

// Pad
ARMNN_AUTO_TEST_CASE(PadFloat322d, PadFloat322dTest)
ARMNN_AUTO_TEST_CASE(PadFloat323d, PadFloat323dTest)
ARMNN_AUTO_TEST_CASE(PadFloat324d, PadFloat324dTest)

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

// Max
ARMNN_AUTO_TEST_CASE(SimpleMaximum, MaximumSimpleTest)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1Element, MaximumBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVector, MaximumBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE(MaximumUint8, MaximumUint8Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1ElementUint8, MaximumBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVectorUint8, MaximumBroadcast1DVectorUint8Test)

// Min
ARMNN_AUTO_TEST_CASE(SimpleMinimum1, MinimumBroadcast1ElementTest1)
ARMNN_AUTO_TEST_CASE(SimpleMinimum2, MinimumBroadcast1ElementTest2)
ARMNN_AUTO_TEST_CASE(Minimum1DVectorUint8, MinimumBroadcast1DVectorUint8Test)

// Normalization
ARMNN_AUTO_TEST_CASE(SimpleNormalizationAcross, SimpleNormalizationAcrossTest)
ARMNN_AUTO_TEST_CASE(SimpleNormalizationWithin, SimpleNormalizationWithinTest)
ARMNN_AUTO_TEST_CASE(SimpleNormalizationAcrossNhwc, SimpleNormalizationAcrossNhwcTest)

// Resize Bilinear - NCHW data layout
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinear, SimpleResizeBilinearTest<armnn::DataType::Float32>, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNop, ResizeBilinearNopTest<armnn::DataType::Float32>, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMin, ResizeBilinearSqMinTest<armnn::DataType::Float32>, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMin, ResizeBilinearMinTest<armnn::DataType::Float32>, armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMag, ResizeBilinearMagTest<armnn::DataType::Float32>, armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearUint8,
                     SimpleResizeBilinearTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopUint8,
                     ResizeBilinearNopTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinUint8,
                     ResizeBilinearSqMinTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinUint8,
                     ResizeBilinearMinTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagUint8,
                     ResizeBilinearMagTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NCHW)

// Resize Bilinear - NHWC data layout
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopNhwc,
                     ResizeBilinearNopTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearNhwc,
                     SimpleResizeBilinearTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinNhwc,
                     ResizeBilinearSqMinTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinNhwc,
                     ResizeBilinearMinTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagNhwc,
                     ResizeBilinearMagTest<armnn::DataType::Float32>,
                     armnn::DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(ResizeBilinearNopUint8Nhwc,
                     ResizeBilinearNopTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearUint8Nhwc,
                     SimpleResizeBilinearTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinUint8Nhwc,
                     ResizeBilinearSqMinTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinUint8Nhwc,
                     ResizeBilinearMinTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagUint8Nhwc,
                     ResizeBilinearMagTest<armnn::DataType::QuantisedAsymm8>,
                     armnn::DataLayout::NHWC)

// Quantize
ARMNN_AUTO_TEST_CASE(QuantizeSimpleUint8, QuantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE(QuantizeClampUint8, QuantizeClampUint8Test)

// ============================================================================
// COMPARE tests

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareConv2dWithReference, CompareConvolution2dTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareDepthwiseConv2dWithReferenceFloat32,
                                 CompareDepthwiseConvolution2dFloatTest,
                                 armnn::DataLayout::NCHW)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareDepthwiseConv2dWithReferenceUint8,
                                 CompareDepthwiseConvolution2dUint8Test,
                                 armnn::DataLayout::NCHW)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareDepthwiseConv2dWithReferenceFloat32Nhwc,
                                 CompareDepthwiseConvolution2dFloatTest,
                                 armnn::DataLayout::NHWC)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareDepthwiseConv2dWithReferenceUint8Nhwc,
                                 CompareDepthwiseConvolution2dUint8Test,
                                 armnn::DataLayout::NHWC)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareNormalizationWithinWithReference, CompareNormalizationTest,
                                 armnn::NormalizationAlgorithmChannel::Within,
                                 armnn::NormalizationAlgorithmMethod::LocalBrightness)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareNormalizationAcrossWithReference, CompareNormalizationTest,
                                 armnn::NormalizationAlgorithmChannel::Across,
                                 armnn::NormalizationAlgorithmMethod::LocalBrightness)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareMaxPooling2dWithReference, ComparePooling2dTest, armnn::PoolingAlgorithm::Max)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareMaxPooling2dWithReferenceUint8, ComparePooling2dUint8Test,
                                 armnn::PoolingAlgorithm::Max)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareAveragePooling2dWithReference, ComparePooling2dTest,
                                 armnn::PoolingAlgorithm::Average)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareAveragePooling2dWithReferenceUint8, ComparePooling2dUint8Test,
                                 armnn::PoolingAlgorithm::Average)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareL2Pooling2dWithReference, ComparePooling2dTest, armnn::PoolingAlgorithm::L2)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(UNSUPPORTED_CompareL2Pooling2dWithReferenceUint8, ComparePooling2dUint8Test,
                                 armnn::PoolingAlgorithm::L2)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareSoftmaxBeta1WithReference, CompareSoftmaxTest, 1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareSoftmaxBeta2WithReference, CompareSoftmaxTest, 2.0f)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareSoftmaxUint8Beta1WithReference, CompareSoftmaxUint8Test, 1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareSoftmaxUint8Beta2WithReference, CompareSoftmaxUint8Test, 2.0f)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareAddition, CompareAdditionTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareMultiplicationWithReference, CompareMultiplicationTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareBatchNorm, CompareBatchNormTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(ReLu1, CompareBoundedReLuTest, 1.0f, -1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(ReLu6, CompareBoundedReLuTest, 6.0f, 0.0f)

// ============================================================================
// FIXTURE tests

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareSigmoidActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::Sigmoid, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareTanhActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::TanH, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareLinearActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::Linear, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareReLuActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::ReLu, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareBoundedReLuActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::BoundedReLu, 5u)
ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareBoundedReLuActivationWithReferenceUint8, ActivationFixture,
                                    CompareActivationUint8Test, armnn::ActivationFunction::BoundedReLu)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareSoftReLuActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::SoftReLu, 1u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareLeakyReLuActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::LeakyReLu, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareAbsActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::Abs, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareSqrtActivationWithReference, PositiveActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::Sqrt, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareSquareActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::Square, 5u)
BOOST_AUTO_TEST_SUITE_END()
