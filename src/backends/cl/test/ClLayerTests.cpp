//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClContextControlFixture.hpp"

#include "test/TensorHelpers.hpp"
#include "test/UnitTests.hpp"

#include <backends/CpuTensorHandle.hpp>
#include <backends/cl/ClLayerSupport.hpp>
#include <backends/cl/ClWorkloadFactory.hpp>
#include <backends/cl/workloads/ClWorkloadUtils.hpp>
#include <backends/reference/RefWorkloadFactory.hpp>
#include <backends/test/ActivationFixture.hpp>
#include <backends/test/LayerTests.hpp>

#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <string>

BOOST_FIXTURE_TEST_SUITE(Compute_ArmComputeCl, ClContextControlFixture)
using FactoryType = armnn::ClWorkloadFactory;

// ============================================================================
// UNIT tests

// Activation
ARMNN_AUTO_TEST_CASE(ConstantLinearActivation, ConstantLinearActivationTest)

ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta1, SimpleSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta2, SimpleSoftmaxTest, 2.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta1Uint8, SimpleSoftmaxUint8Test, 1.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta2Uint8, SimpleSoftmaxUint8Test, 2.0f)

ARMNN_AUTO_TEST_CASE(ReLu1Uint8, BoundedReLuUint8UpperAndLowerBoundTest)
ARMNN_AUTO_TEST_CASE(ReLu6Uint8, BoundedReLuUint8UpperBoundOnlyTest)

// Fully Connected
ARMNN_AUTO_TEST_CASE(SimpleFullyConnected, FullyConnectedFloat32Test, false, false)
ARMNN_AUTO_TEST_CASE(SimpleFullyConnectedWithBias, FullyConnectedFloat32Test, true, false)
ARMNN_AUTO_TEST_CASE(SimpleFullyConnectedWithTranspose, FullyConnectedFloat32Test, false, true)
ARMNN_AUTO_TEST_CASE(FullyConnectedUint8, FullyConnectedUint8Test, false)
ARMNN_AUTO_TEST_CASE(FullyConnectedBiasedUint8, FullyConnectedUint8Test, true)

ARMNN_AUTO_TEST_CASE(FullyConnectedLarge, FullyConnectedLargeTest, false)
ARMNN_AUTO_TEST_CASE(FullyConnectedLargeTransposed, FullyConnectedLargeTest, true)

// Convolution
ARMNN_AUTO_TEST_CASE(SimpleConvolution1d, Convolution1dTest, true)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2d, SimpleConvolution2d3x5Test, true)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2dSquare, SimpleConvolution2d3x3Test, true)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2d3x3Uint8, SimpleConvolution2d3x3Uint8Test, true)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2d, SimpleConvolution2d3x5Test, false)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dSquare, SimpleConvolution2d3x3Test, false)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2dAsymmetricPadding, Convolution2dAsymmetricPaddingTest)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2dSquareNhwc, SimpleConvolution2d3x3NhwcTest, false)

// Depthwise Convolution
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1, DepthwiseConvolution2dDepthMul1Test, true)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthNhwc, DepthwiseConvolution2dDepthNhwcTest, false)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1, DepthwiseConvolution2dDepthMul1Test, false)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1Uint8, DepthwiseConvolution2dDepthMul1Uint8Test, true)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1Uint8, DepthwiseConvolution2dDepthMul1Uint8Test, false)

ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dAsymmetric, DepthwiseConvolution2dAsymmetricTest, true)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dAsymmetric, DepthwiseConvolution2dAsymmetricTest, false)

// Softmax
BOOST_AUTO_TEST_CASE(Softmax4dSupport)
{
    const unsigned int numDimensions = 4u;
    std::array<unsigned int, numDimensions> dimensionSizes;
    dimensionSizes.fill(1u);

    const armnn::TensorInfo inputInfo(numDimensions, &dimensionSizes.front(), armnn::DataType::Float32);
    const armnn::TensorInfo outputInfo(numDimensions, &dimensionSizes.front(), armnn::DataType::Float32);

    // 4D Softmax should be reported as unsupported on the CL backend
    BOOST_TEST(!armnn::IsSoftmaxSupportedCl(inputInfo, outputInfo, armnn::SoftmaxDescriptor()));
}

// Splitter
ARMNN_AUTO_TEST_CASE(SimpleSplitter, SplitterTest)
ARMNN_AUTO_TEST_CASE(SimpleSplitterUint8, SplitterUint8Test)

ARMNN_AUTO_TEST_CASE(CopyViaSplitter, CopyViaSplitterTest)
ARMNN_AUTO_TEST_CASE(CopyViaSplitterUint8, CopyViaSplitterUint8Test)

// Merger
ARMNN_AUTO_TEST_CASE(SimpleMerger, MergerTest)
ARMNN_AUTO_TEST_CASE(MergerUint8, MergerUint8Test)

// Normalization
ARMNN_AUTO_TEST_CASE(SimpleNormalizationAcross, SimpleNormalizationAcrossTest)
ARMNN_AUTO_TEST_CASE(SimpleNormalizationWithin, SimpleNormalizationWithinTest)
ARMNN_AUTO_TEST_CASE(SimpleNormalizationAcrossNhwc, SimpleNormalizationAcrossNhwcTest)

// Pooling
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4, SimpleMaxPooling2dSize3x3Stride2x4Test, true)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4Uint8, SimpleMaxPooling2dSize3x3Stride2x4Uint8Test, true)

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

ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleL2Pooling2d, IgnorePaddingSimpleL2Pooling2dTest)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_IgnorePaddingSimpleL2Pooling2dUint8, IgnorePaddingSimpleL2Pooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingL2Pooling2dSize3, IgnorePaddingL2Pooling2dSize3Test)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_IgnorePaddingL2Pooling2dSize3Uint8, IgnorePaddingL2Pooling2dSize3Uint8Test)

ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2d, SimpleAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dNhwc, SimpleAveragePooling2dNhwcTest)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dUint8, SimpleAveragePooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3x2Stride2x2,
                     IgnorePaddingAveragePooling2dSize3x2Stride2x2Test,
                     false)
ARMNN_AUTO_TEST_CASE(IgnorePaddingAveragePooling2dSize3x2Stride2x2NoPadding,
                     IgnorePaddingAveragePooling2dSize3x2Stride2x2Test,
                     true)
ARMNN_AUTO_TEST_CASE(LargeTensorsAveragePooling2d, LargeTensorsAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE(LargeTensorsAveragePooling2dUint8, LargeTensorsAveragePooling2dUint8Test)

ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2d, SimpleL2Pooling2dTest)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_SimpleL2Pooling2dUint8, SimpleL2Pooling2dUint8Test)
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

// Add
ARMNN_AUTO_TEST_CASE(SimpleAdd, AdditionTest)
ARMNN_AUTO_TEST_CASE(AddBroadcast1Element, AdditionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(AddBroadcast, AdditionBroadcastTest)

ARMNN_AUTO_TEST_CASE(AdditionUint8, AdditionUint8Test)
ARMNN_AUTO_TEST_CASE(AddBroadcastUint8, AdditionBroadcastUint8Test)
ARMNN_AUTO_TEST_CASE(AddBroadcast1ElementUint8, AdditionBroadcast1ElementUint8Test)

// Sub
ARMNN_AUTO_TEST_CASE(SimpleSub, SubtractionTest)

// Div
ARMNN_AUTO_TEST_CASE(SimpleDivision, DivisionTest)
ARMNN_AUTO_TEST_CASE(DivisionByZero, DivisionByZeroTest)
ARMNN_AUTO_TEST_CASE(DivisionBroadcast1Element, DivisionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(DivisionBroadcast1DVector, DivisionBroadcast1DVectorTest)
// NOTE: quantized division is not supported by CL and not required by the
//       android NN api

// Mul
ARMNN_AUTO_TEST_CASE(SimpleMultiplication, MultiplicationTest)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1Element, MultiplicationBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1DVector, MultiplicationBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE(MultiplicationUint8, MultiplicationUint8Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1ElementUint8, MultiplicationBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1DVectorUint8, MultiplicationBroadcast1DVectorUint8Test)

// Batch Norm
ARMNN_AUTO_TEST_CASE(BatchNorm, BatchNormTest)

// L2 Normalization
ARMNN_AUTO_TEST_CASE(L2Normalization1d, L2Normalization1dTest)
ARMNN_AUTO_TEST_CASE(L2Normalization2d, L2Normalization2dTest)
ARMNN_AUTO_TEST_CASE(L2Normalization3d, L2Normalization3dTest)
ARMNN_AUTO_TEST_CASE(L2Normalization4d, L2Normalization4dTest)

// NOTE: The following test hits a bug in ACL that makes it fail, keep it disabled until a patch is available in ACL
//ARMNN_AUTO_TEST_CASE(L2Normalization1dNhwc, L2Normalization1dNhwcTest)
ARMNN_AUTO_TEST_CASE(L2Normalization2dNhwc, L2Normalization2dNhwcTest)
ARMNN_AUTO_TEST_CASE(L2Normalization3dNhwc, L2Normalization3dNhwcTest)
ARMNN_AUTO_TEST_CASE(L2Normalization4dNhwc, L2Normalization4dNhwcTest)

// Resize Bilinear - NCHW data layout
ARMNN_AUTO_TEST_CASE(ResizeBilinearNop, ResizeBilinearNopTest)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinear, SimpleResizeBilinearTest)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMin, ResizeBilinearSqMinTest)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMin, ResizeBilinearMinTest)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMag, ResizeBilinearMagTest)

// Resize Bilinear - NHWC data layout
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopNhwc, ResizeBilinearNopNhwcTest)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearNhwc, SimpleResizeBilinearNhwcTest)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinNhwc, ResizeBilinearSqMinNhwcTest)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinNhwc, ResizeBilinearMinNhwcTest)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagNhwc, ResizeBilinearMagNhwcTest)

// Constant
ARMNN_AUTO_TEST_CASE(Constant, ConstantTest)
ARMNN_AUTO_TEST_CASE(ConstantUint8, ConstantTestUint8)

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
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2, Concatenation3dDim2Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2Uint8, Concatenation3dDim2Uint8Test)

ARMNN_AUTO_TEST_CASE(Concatenation3dDim0DiffInputDims, Concatenation3dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim0DiffInputDimsUint8, Concatenation3dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1DiffInputDims, Concatenation3dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1DiffInputDimsUint8, Concatenation3dDim1DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2DiffInputDims, Concatenation3dDim2DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2DiffInputDimsUint8, Concatenation3dDim2DiffInputDimsUint8Test)

// Floor
ARMNN_AUTO_TEST_CASE(SimpleFloor, SimpleFloorTest)

// Reshape
ARMNN_AUTO_TEST_CASE(SimpleReshapeFloat32, SimpleReshapeFloat32Test)
ARMNN_AUTO_TEST_CASE(SimpleReshapeUint8, SimpleReshapeUint8Test)

// Pad
ARMNN_AUTO_TEST_CASE(Pad2d, Pad2dTest)
ARMNN_AUTO_TEST_CASE(Pad3d, Pad3dTest)
ARMNN_AUTO_TEST_CASE(Pad4d, Pad4dTest)

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

// Convert from Float16 to Float32
ARMNN_AUTO_TEST_CASE(SimpleConvertFp16ToFp32, SimpleConvertFp16ToFp32Test)
// Convert from Float32 to Float16
ARMNN_AUTO_TEST_CASE(SimpleConvertFp32ToFp16, SimpleConvertFp32ToFp16Test)

ARMNN_AUTO_TEST_CASE(AdditionAfterMaxPool, AdditionAfterMaxPoolTest)

// ============================================================================
// COMPARE tests

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareConv2dWithReference, CompareConvolution2dTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareDepthwiseConv2dWithReferenceFloat32, CompareDepthwiseConvolution2dTest<float>)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareDepthwiseConv2dWithReferenceUint8, CompareDepthwiseConvolution2dTest<uint8_t>)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareNormalizationWithinWithReference, CompareNormalizationTest,
                                 armnn::NormalizationAlgorithmChannel::Within,
                                 armnn::NormalizationAlgorithmMethod::LocalBrightness)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareNormalizationAcrossWithReference, CompareNormalizationTest,
                                 armnn::NormalizationAlgorithmChannel::Across,
                                 armnn::NormalizationAlgorithmMethod::LocalBrightness)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareSoftmaxBeta1WithReference, CompareSoftmaxTest, 1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareSoftmaxBeta2WithReference, CompareSoftmaxTest, 2.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareSoftmaxUint8, CompareSoftmaxUint8Test, 1.0f)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareMaxPooling2dWithRef, ComparePooling2dTest, armnn::PoolingAlgorithm::Max)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareAveragePooling2dWithRef, ComparePooling2dTest, armnn::PoolingAlgorithm::Average)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareAveragePooling2dWithRefUint8, ComparePooling2dUint8Test,
                                 armnn::PoolingAlgorithm::Average)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareL2Pooling2dWithRef, ComparePooling2dTest, armnn::PoolingAlgorithm::L2)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareAddition, CompareAdditionTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareMultiplicationWithRef, CompareMultiplicationTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareBatchNorm, CompareBatchNormTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareReLu1, CompareBoundedReLuTest, 1.0f, -1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE(CompareReLu6, CompareBoundedReLuTest, 6.0f, 0.0f)

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
                                    CompareActivationTest, armnn::ActivationFunction::SoftReLu, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareLeakyReLuActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::LeakyReLu, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareAbsActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::Abs, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareSqrtActivationWithReference, PositiveActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::Sqrt, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE(CompareSquareActivationWithReference, ActivationFixture,
                                    CompareActivationTest, armnn::ActivationFunction::Square, 5u)

BOOST_AUTO_TEST_SUITE_END()
