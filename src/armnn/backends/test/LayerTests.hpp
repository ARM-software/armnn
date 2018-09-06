//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#pragma once

#include "armnn/ArmNN.hpp"
#include "armnn/Tensor.hpp"
#include "Half.hpp"

#include <boost/multi_array.hpp>
#include <boost/assert.hpp>
#include <array>

// Layer callables.

namespace armnn
{
class IWorkloadFactory;
}

template <std::size_t n>
boost::array<unsigned int, n> GetTensorShapeAsArray(const armnn::TensorInfo& tensorInfo)
{
    BOOST_ASSERT_MSG(n == tensorInfo.GetNumDimensions(),
        "Attempting to construct a shape array of mismatching size");

    boost::array<unsigned int, n> shape;
    for (unsigned int i = 0; i < n; i++)
    {
        shape[i] = tensorInfo.GetShape()[i];
    }
    return shape;
}

template <typename T, std::size_t n>
struct LayerTestResult
{
    LayerTestResult(const armnn::TensorInfo& outputInfo)
    {
        auto shape( GetTensorShapeAsArray<n>(outputInfo) );
        output.resize(shape);
        outputExpected.resize(shape);
        supported = true;
    }

    boost::multi_array<T, n> output;
    boost::multi_array<T, n> outputExpected;
    bool supported;
};

LayerTestResult<float, 4> SimpleConvolution2d3x5Test(armnn::IWorkloadFactory& workloadFactory,
                                                     bool                     biasEnabled);

LayerTestResult<float, 4> SimpleConvolution2d3x3Test(armnn::IWorkloadFactory& workloadFactory,
                                                     bool                     biasEnabled);

LayerTestResult<float, 4>
Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> Convolution2dAsymmetricPaddingTest(armnn::IWorkloadFactory& workloadFactory);


LayerTestResult<float,   4> Convolution1dTest(armnn::IWorkloadFactory& workloadFactory, bool biasEnabled);
LayerTestResult<uint8_t, 4> Convolution1dUint8Test(armnn::IWorkloadFactory& workloadFactory, bool biasEnabled);

LayerTestResult<float, 4> DepthwiseConvolution2dTest(armnn::IWorkloadFactory& workloadFactory, bool biasEnabled);

LayerTestResult<float, 4> DepthwiseConvolution2dDepthMul1Test(armnn::IWorkloadFactory& workloadFactory,
                                                              bool biasEnabled);

LayerTestResult<float, 4> DepthwiseConvolution2dAsymmetricTest(armnn::IWorkloadFactory& workloadFactory,
                                                               bool biasEnabled);

LayerTestResult<float,   4> SimpleMaxPooling2dSize2x2Stride2x2Test(armnn::IWorkloadFactory& workloadFactory,
                                                                   bool forceNoPadding);
LayerTestResult<uint8_t, 4> SimpleMaxPooling2dSize2x2Stride2x2Uint8Test(armnn::IWorkloadFactory& workloadFactory,
                                                                        bool forceNoPadding);
LayerTestResult<float,   4> SimpleMaxPooling2dSize3x3Stride2x4Test(armnn::IWorkloadFactory& workloadFactory,
                                                                   bool forceNoPadding);
LayerTestResult<uint8_t, 4> SimpleMaxPooling2dSize3x3Stride2x4Uint8Test(armnn::IWorkloadFactory& workloadFactory,
                                                                        bool forceNoPadding );
LayerTestResult<float,   4> IgnorePaddingSimpleMaxPooling2dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> IgnorePaddingSimpleMaxPooling2dUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float,   4> IgnorePaddingMaxPooling2dSize3Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> IgnorePaddingMaxPooling2dSize3Uint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float,   4> SimpleAveragePooling2dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> SimpleAveragePooling2dUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float,   4> IgnorePaddingAveragePooling2dSize3x2Stride2x2Test(armnn::IWorkloadFactory& workloadFactory,
                                                                              bool forceNoPadding);
LayerTestResult<float,   4> IgnorePaddingSimpleAveragePooling2dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> IgnorePaddingSimpleAveragePooling2dUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4>   IgnorePaddingSimpleAveragePooling2dNoPaddingTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> IgnorePaddingSimpleAveragePooling2dNoPaddingUint8Test(
    armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float,   4> IgnorePaddingAveragePooling2dSize3Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> IgnorePaddingAveragePooling2dSize3Uint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float,   4> SimpleL2Pooling2dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> SimpleL2Pooling2dUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float,   4> L2Pooling2dSize3Stride1Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride1Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float,   4> L2Pooling2dSize3Stride3Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride3Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float,   4> L2Pooling2dSize3Stride4Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> L2Pooling2dSize3Stride4Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float,   4> L2Pooling2dSize7Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> L2Pooling2dSize7Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float,   4> L2Pooling2dSize9Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> L2Pooling2dSize9Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float,   4> LargeTensorsAveragePooling2dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> LargeTensorsAveragePooling2dUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float,   4> IgnorePaddingSimpleL2Pooling2dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> IgnorePaddingSimpleL2Pooling2dUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float,   4> IgnorePaddingL2Pooling2dSize3Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> IgnorePaddingL2Pooling2dSize3Uint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float,   4> AsymmetricNonSquarePooling2dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> AsymmetricNonSquarePooling2dUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> ComparePooling2dTest(armnn::IWorkloadFactory& workloadFactory,
                                               armnn::IWorkloadFactory& refWorkloadFactory,
                                               armnn::PoolingAlgorithm  poolingType);
LayerTestResult<uint8_t, 4> ComparePooling2dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                                      armnn::IWorkloadFactory& refWorkloadFactory,
                                                      armnn::PoolingAlgorithm  poolingType);

LayerTestResult<float, 4> ConstantLinearActivationTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> SimpleNormalizationAcrossTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> SimpleNormalizationWithinTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 2> SimpleSoftmaxTest(armnn::IWorkloadFactory& workloadFactory, float beta);
LayerTestResult<uint8_t, 2> SimpleSoftmaxUint8Test(armnn::IWorkloadFactory& workloadFactory, float beta);

LayerTestResult<float, 4> SimpleSigmoidTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> SimpleReshapeFloat32Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> SimpleReshapeUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> SimpleFloorTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 1> Concatenation1dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 2> Concatenation2dDim0Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 2> Concatenation2dDim1Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 2> Concatenation2dDim0DiffInputDimsTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 2> Concatenation2dDim1DiffInputDimsTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 3> Concatenation3dDim0Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 3> Concatenation3dDim1Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 3> Concatenation3dDim2Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 3> Concatenation3dDim0DiffInputDimsTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 3> Concatenation3dDim1DiffInputDimsTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 3> Concatenation3dDim2DiffInputDimsTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> SimpleSigmoidUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> CompareConvolution2dTest(armnn::IWorkloadFactory& workloadFactory,
    armnn::IWorkloadFactory& refWorkloadFactory);

template<typename T>
LayerTestResult<T, 4> CompareDepthwiseConvolution2dTest(armnn::IWorkloadFactory& workloadFactory,
    armnn::IWorkloadFactory& refWorkloadFactory);

LayerTestResult<float, 4> CompareNormalizationTest(armnn::IWorkloadFactory& workloadFactory,
                                                   armnn::IWorkloadFactory& refWorkloadFactory,
                                                   armnn::NormalizationAlgorithmChannel normChannel,
                                                   armnn::NormalizationAlgorithmMethod normMethod);

LayerTestResult<float, 2> CompareSoftmaxTest(armnn::IWorkloadFactory& workloadFactory,
    armnn::IWorkloadFactory& refWorkloadFactory, float beta);

LayerTestResult<float, 2> FullyConnectedFloat32Test(armnn::IWorkloadFactory& workloadFactory,
                                             bool                     biasEnabled,
                                             bool                     transposeWeights);

std::vector<LayerTestResult<float, 3>> SplitterTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 3> CopyViaSplitterTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 3> MergerTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> AdditionTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> AdditionBroadcast1ElementTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> AdditionBroadcastTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> CompareAdditionTest(armnn::IWorkloadFactory& workloadFactory,
                                              armnn::IWorkloadFactory& refWorkloadFactory);

LayerTestResult<float, 4> SubtractionTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> SubtractionBroadcast1ElementTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> SubtractionBroadcastTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> CompareActivationTest(armnn::IWorkloadFactory&  workloadFactory,
                                                armnn::IWorkloadFactory&  refWorkloadFactory,
                                                armnn::ActivationFunction f,
                                                unsigned int batchSize);

LayerTestResult<float, 4> DivisionTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> DivisionByZeroTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> DivisionBroadcast1ElementTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> DivisionBroadcast1DVectorTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> MultiplicationTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> MultiplicationBroadcast1ElementTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> MultiplicationBroadcast1DVectorTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> CompareMultiplicationTest(armnn::IWorkloadFactory& workloadFactory,
                                             armnn::IWorkloadFactory& refWorkloadFactory);

LayerTestResult<float, 4> BatchNormTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> CompareBatchNormTest(armnn::IWorkloadFactory& workloadFactory,
                                        armnn::IWorkloadFactory& refWorkloadFactory);

LayerTestResult<float, 4> BoundedReLuUpperAndLowerBoundTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> BoundedReLuUint8UpperAndLowerBoundTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> BoundedReLuUpperBoundOnlyTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> BoundedReLuUint8UpperBoundOnlyTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> CompareBoundedReLuTest(armnn::IWorkloadFactory& workloadFactory,
                                                 armnn::IWorkloadFactory& refWorkloadFactory,
                                                 float upperBound,
                                                 float lowerBound);

// Tests that the output should be identical to the input when the output dimensions match the input ones.
LayerTestResult<float, 4> ResizeBilinearNopTest(armnn::IWorkloadFactory& workloadFactory);

// Tests the behaviour of the resize bilinear operation when rescaling a 2x2 image into a 1x1 image.
LayerTestResult<float, 4> SimpleResizeBilinearTest(armnn::IWorkloadFactory& workloadFactory);

// Tests the resize bilinear for minification of a square input matrix (also: input dimensions are a
// multiple of output dimensions).
LayerTestResult<float, 4> ResizeBilinearSqMinTest(armnn::IWorkloadFactory& workloadFactory);

// Tests the resize bilinear for minification (output dimensions smaller than input dimensions).
LayerTestResult<float, 4> ResizeBilinearMinTest(armnn::IWorkloadFactory& workloadFactory);

// Tests the resize bilinear for magnification (output dimensions bigger than input dimensions).
LayerTestResult<float, 4> ResizeBilinearMagTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> BatchNormTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 2> FakeQuantizationTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> L2Normalization1dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> L2Normalization2dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> L2Normalization3dTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> L2Normalization4dTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> ConstantTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> ConstantTestUint8(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> BoundedReLuUint8Test(armnn::IWorkloadFactory& workloadFactory, float upperBound);
LayerTestResult<uint8_t, 4> BoundedReLuUint8Test(armnn::IWorkloadFactory& workloadFactory,
    float upperBound,
    float lowerBound);

LayerTestResult<uint8_t, 2> FullyConnectedUint8Test(armnn::IWorkloadFactory& workloadFactory, bool biasEnabled);

std::vector<LayerTestResult<uint8_t, 3>> SplitterUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 3> CopyViaSplitterUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 3> MergerUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> AdditionUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> AdditionBroadcast1ElementUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> AdditionBroadcastUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> SubtractionUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> SubtractionBroadcast1ElementUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> SubtractionBroadcastUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> CompareActivationUint8Test(armnn::IWorkloadFactory&  workloadFactory,
                                                       armnn::IWorkloadFactory&  refWorkloadFactory,
                                                       armnn::ActivationFunction f);

LayerTestResult<uint8_t, 2> CompareSoftmaxUint8Test(armnn::IWorkloadFactory& workloadFactory,
    armnn::IWorkloadFactory& refWorkloadFactory,
    float beta);

LayerTestResult<uint8_t, 4> MultiplicationUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> MultiplicationBroadcast1ElementUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> MultiplicationBroadcast1DVectorUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x5Uint8Test(armnn::IWorkloadFactory& workloadFactory,
                                                            bool                     biasEnabled);

LayerTestResult<uint8_t, 4> SimpleConvolution2d3x3Uint8Test(armnn::IWorkloadFactory& workloadFactory,
                                                            bool                     biasEnabled);

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dUint8Test(armnn::IWorkloadFactory& workloadFactory,
                                                            bool                     biasEnabled);

LayerTestResult<uint8_t, 4> DepthwiseConvolution2dDepthMul1Uint8Test(armnn::IWorkloadFactory& workloadFactory,
                                                                     bool biasEnabled);

LayerTestResult<uint8_t, 4> ConstantLinearActivationUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> ResizeBilinearNopUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> SimpleResizeBilinearUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> ResizeBilinearSqMinUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> ResizeBilinearMinUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> ResizeBilinearMagUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> BatchNormUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 4> ConstantUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<uint8_t, 1> Concatenation1dUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 2> Concatenation2dDim0Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 2> Concatenation2dDim1Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 2> Concatenation2dDim0DiffInputDimsUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 2> Concatenation2dDim1DiffInputDimsUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 3> Concatenation3dDim0Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 3> Concatenation3dDim1Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 3> Concatenation3dDim2Uint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 3> Concatenation3dDim0DiffInputDimsUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 3> Concatenation3dDim1DiffInputDimsUint8Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 3> Concatenation3dDim2DiffInputDimsUint8Test(armnn::IWorkloadFactory& workloadFactory);


LayerTestResult<float, 2> FullyConnectedLargeTest(armnn::IWorkloadFactory& workloadFactory,
                                                  bool transposeWeights);
LayerTestResult<float, 4> SimplePermuteFloat32Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<uint8_t, 4> SimplePermuteUint8Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> PermuteFloat32ValueSet1Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> PermuteFloat32ValueSet2Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 4> PermuteFloat32ValueSet3Test(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 2> LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest
        (armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 2>
        LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<float, 2>
LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest(armnn::IWorkloadFactory& workloadFactory);

LayerTestResult<float, 4> SimpleConvertFp16ToFp32Test(armnn::IWorkloadFactory& workloadFactory);
LayerTestResult<armnn::Half, 4> SimpleConvertFp32ToFp16Test(armnn::IWorkloadFactory& workloadFactory);
