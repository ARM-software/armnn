//
// Copyright © 2017 Arm Ltd. All rights reserved.
// See LICENSE file in the project root for full license information.
//
#include <boost/test/unit_test.hpp>

#include "test/TensorHelpers.hpp"
#include "LayerTests.hpp"

#include "backends/CpuTensorHandle.hpp"
#include "backends/NeonLayerSupport.hpp"
#include "backends/NeonWorkloadFactory.hpp"
#include "backends/RefWorkloadFactory.hpp"
#include "backends/test/TensorCopyUtils.hpp"
#include "ActivationFixture.hpp"

#include "WorkloadTestUtils.hpp"

#include "test/UnitTests.hpp"

BOOST_AUTO_TEST_SUITE(Compute_ArmComputeNeon)
using FactoryType = armnn::NeonWorkloadFactory;

// ============================================================================
// UNIT tests

// Convolution
ARMNN_AUTO_TEST_CASE(SimpleConvolution1d, Convolution1dTest, true)

ARMNN_AUTO_TEST_CASE(SimpleConvolution2d, SimpleConvolution2d3x5Test, true)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2dSquare, SimpleConvolution2d3x3Test, true)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2d, SimpleConvolution2d3x5Test, false)
ARMNN_AUTO_TEST_CASE(UnbiasedConvolution2dSquare, SimpleConvolution2d3x3Test, false)
ARMNN_AUTO_TEST_CASE(SimpleConvolution2dAsymmetricPadding, Convolution2dAsymmetricPaddingTest)

namespace
{

armnn::Convolution2dDescriptor MakeConv2dDesc(uint32_t strideX, uint32_t strideY,
    uint32_t padLeft = 0, uint32_t padRight = 0, uint32_t padTop = 0, uint32_t padBottom = 0)
{
    armnn::Convolution2dDescriptor result;
    result.m_StrideX = strideX;
    result.m_StrideY = strideY;
    result.m_PadLeft = padLeft;
    result.m_PadRight = padRight;
    result.m_PadTop = padTop;
    result.m_PadBottom = padBottom;
    result.m_BiasEnabled = true;
    return result;
}

}

BOOST_AUTO_TEST_CASE(Conv2dUtils)
{
    // the only preferred Neon convolution is 1x1 with padding=0 and stride size {1,2,3}
    armnn::TensorShape shape1x1({ 1,1,1,1 });
    armnn::TensorInfo info1x1(shape1x1, armnn::DataType::Float32);
    BOOST_TEST(armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(1, 1)));
    BOOST_TEST(armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(1, 2)));
    BOOST_TEST(armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(1, 3)));
    BOOST_TEST(armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(2, 1)));
    BOOST_TEST(armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(2, 2)));
    BOOST_TEST(armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(2, 3)));
    BOOST_TEST(armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(3, 1)));
    BOOST_TEST(armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(3, 2)));
    BOOST_TEST(armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(3, 3)));

    BOOST_TEST(!armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(4, 1)));
    BOOST_TEST(!armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(4, 5)));
    BOOST_TEST(!armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(3, 6)));

    // non zero padding is not preferred for direct convolution
    BOOST_TEST(!armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(1, 1, 1, 0)));
    BOOST_TEST(!armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(1, 1, 0, 1)));
    BOOST_TEST(!armnn::IsNeonDirectConvolutionPreferred(info1x1, MakeConv2dDesc(1, 1, 1, 1)));

    // 2x2 filter not preferred for direct convolution
    armnn::TensorShape shape2x2({ 1,1,2,2 });
    armnn::TensorInfo info2x2(shape2x2, armnn::DataType::Float32);
    BOOST_TEST(!armnn::IsNeonDirectConvolutionPreferred(info2x2, MakeConv2dDesc(1, 1)));
}

// Depthwise Convolution
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1, DepthwiseConvolution2dDepthMul1Test, true)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1, DepthwiseConvolution2dDepthMul1Test, false)
ARMNN_AUTO_TEST_CASE(DepthwiseConvolution2dDepthMul1Uint8, DepthwiseConvolution2dDepthMul1Uint8Test, true)
ARMNN_AUTO_TEST_CASE(UnbiasedDepthwiseConvolution2dDepthMul1Uint8, DepthwiseConvolution2dDepthMul1Uint8Test, false)

namespace
{

armnn::DepthwiseConvolution2dDescriptor MakeDepthwiseConv2dDesc(uint32_t strideX, uint32_t strideY,
    uint32_t depthMultiplier = 1, uint32_t padLeft = 0, uint32_t padRight = 0,
    uint32_t padTop = 0, uint32_t padBottom = 0)
{
    armnn::DepthwiseConvolution2dDescriptor desc;
    desc.m_PadLeft = padLeft;
    desc.m_PadRight = padRight;
    desc.m_PadTop = padTop;
    desc.m_PadBottom = padBottom;
    desc.m_StrideX = strideX;
    desc.m_StrideY = strideY;
    desc.m_BiasEnabled = true;
    return desc;
}

}

BOOST_AUTO_TEST_CASE(DepthwiseConv2dUtils)
{
    armnn::TensorInfo inputInfo({ 1, 1, 10, 10 }, armnn::DataType::Float32);
    armnn::TensorInfo weightsInfo3x3({ 1, 1, 3, 3 }, armnn::DataType::Float32);

    // Strides supported: 1,2,3
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(1, 1), weightsInfo3x3));
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(1, 2), weightsInfo3x3));
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(1, 3), weightsInfo3x3));
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(2, 1), weightsInfo3x3));
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(2, 2), weightsInfo3x3));
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(2, 3), weightsInfo3x3));
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(3, 1), weightsInfo3x3));
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(3, 2), weightsInfo3x3));
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(3, 3), weightsInfo3x3));

    // Unsupported stride
    BOOST_TEST(!armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(4, 1), weightsInfo3x3));

    // Supported weights shape 1x1
    armnn::TensorInfo weightsInfo1x1({ 1, 1, 1, 1 }, armnn::DataType::Float32);
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(1, 1), weightsInfo1x1));

    // Supported shape 2x2
    armnn::TensorInfo weightsInfo2x2({ 1, 1, 2, 2 }, armnn::DataType::Float32);
    BOOST_TEST(armnn::IsDepthwiseConvolutionSupportedNeon(inputInfo, MakeDepthwiseConv2dDesc(1, 1), weightsInfo2x2));
}

// Pooling
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4, SimpleMaxPooling2dSize3x3Stride2x4Test, true)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4Uint8, SimpleMaxPooling2dSize3x3Stride2x4Uint8Test, true)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2d, SimpleAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dUint8, SimpleAveragePooling2dUint8Test)
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

ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleL2Pooling2d, IgnorePaddingSimpleL2Pooling2dTest)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_IgnorePaddingSimpleL2Pooling2dUint8, IgnorePaddingSimpleL2Pooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingL2Pooling2dSize3, IgnorePaddingL2Pooling2dSize3Test)
ARMNN_AUTO_TEST_CASE(UNSUPPORTED_IgnorePaddingL2Pooling2dSize3Uint8, IgnorePaddingL2Pooling2dSize3Uint8Test)

// Activation
ARMNN_AUTO_TEST_CASE(ConstantLinearActivation, ConstantLinearActivationTest)

ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta1, SimpleSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta2, SimpleSoftmaxTest, 2.0f)

ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta1Uint8, SimpleSoftmaxUint8Test, 1.0f)
ARMNN_AUTO_TEST_CASE(SimpleSoftmaxBeta2Uint8, SimpleSoftmaxUint8Test, 2.0f)

ARMNN_AUTO_TEST_CASE(ReLu1Uint8, BoundedReLuUint8UpperAndLowerBoundTest)
ARMNN_AUTO_TEST_CASE(ReLu6Uint8, BoundedReLuUint8UpperBoundOnlyTest)

// Splitter
BOOST_AUTO_TEST_CASE(SimpleSplitter)
{
    armnn::NeonWorkloadFactory workloadFactory;
    auto testResult = SplitterTest(workloadFactory);
    for (unsigned int i = 0; i < testResult.size(); ++i)
    {
        BOOST_TEST(CompareTensors(testResult[i].output, testResult[i].outputExpected));
    }
}

BOOST_AUTO_TEST_CASE(SimpleSplitterUint8)
{
    armnn::NeonWorkloadFactory workloadFactory;
    auto testResult = SplitterUint8Test(workloadFactory);
    for (unsigned int i = 0; i < testResult.size(); ++i)
    {
        BOOST_TEST(CompareTensors(testResult[i].output, testResult[i].outputExpected));
    }
}

ARMNN_AUTO_TEST_CASE(CopyViaSplitter, CopyViaSplitterTest)
ARMNN_AUTO_TEST_CASE(CopyViaSplitterUint8, CopyViaSplitterUint8Test)

// Merger
ARMNN_AUTO_TEST_CASE(SimpleMerger, MergerTest)
ARMNN_AUTO_TEST_CASE(MergerUint8, MergerUint8Test)

// Fully Connected
ARMNN_AUTO_TEST_CASE(SimpleFullyConnected, FullyConnectedFloat32Test, false, false)
ARMNN_AUTO_TEST_CASE(SimpleFullyConnectedWithBias, FullyConnectedFloat32Test, true, false)
ARMNN_AUTO_TEST_CASE(SimpleFullyConnectedWithTranspose, FullyConnectedFloat32Test, false, true)
ARMNN_AUTO_TEST_CASE(FullyConnectedLarge, FullyConnectedLargeTest, false)
ARMNN_AUTO_TEST_CASE(FullyConnectedLargeTransposed, FullyConnectedLargeTest, true)

// Add
ARMNN_AUTO_TEST_CASE(SimpleAdd, AdditionTest)
ARMNN_AUTO_TEST_CASE(AddBroadcast1Element, AdditionBroadcast1ElementTest)

// Mul
ARMNN_AUTO_TEST_CASE(SimpleMultiplication, MultiplicationTest)

// Batch Norm
ARMNN_AUTO_TEST_CASE(BatchNorm, BatchNormTest)

// Constant
ARMNN_AUTO_TEST_CASE(Constant, ConstantTest)
ARMNN_AUTO_TEST_CASE(ConstantUint8, ConstantTestUint8)

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
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2, Concatenation3dDim2Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2Uint8, Concatenation3dDim2Uint8Test)

ARMNN_AUTO_TEST_CASE(Concatenation3dDim0DiffInputDims, Concatenation3dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim0DiffInputDimsUint8, Concatenation3dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1DiffInputDims, Concatenation3dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim1DiffInputDimsUint8, Concatenation3dDim1DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2DiffInputDims, Concatenation3dDim2DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE(Concatenation3dDim2DiffInputDimsUint8, Concatenation3dDim2DiffInputDimsUint8Test)

// L2 Normalization
ARMNN_AUTO_TEST_CASE(L2Normalization1d, L2Normalization1dTest);
ARMNN_AUTO_TEST_CASE(L2Normalization2d, L2Normalization2dTest);
ARMNN_AUTO_TEST_CASE(L2Normalization3d, L2Normalization3dTest);
ARMNN_AUTO_TEST_CASE(L2Normalization4d, L2Normalization4dTest);

// Floor
ARMNN_AUTO_TEST_CASE(SimpleFloor, SimpleFloorTest)

// Reshape
ARMNN_AUTO_TEST_CASE(SimpleReshapeFloat32, SimpleReshapeFloat32Test)
ARMNN_AUTO_TEST_CASE(SimpleReshapeUint8, SimpleReshapeUint8Test)

// Permute
ARMNN_AUTO_TEST_CASE(SimplePermuteFloat32, SimplePermuteFloat32Test)
ARMNN_AUTO_TEST_CASE(SimplePermuteUint8, SimplePermuteUint8Test)
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
