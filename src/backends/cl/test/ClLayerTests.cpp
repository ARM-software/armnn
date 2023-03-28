//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ClContextControlFixture.hpp"
#include "ClWorkloadFactoryHelper.hpp"

#include <armnnTestUtils/TensorHelpers.hpp>
#include <UnitTests.hpp>

#include <cl/ClLayerSupport.hpp>
#include <cl/ClWorkloadFactory.hpp>
#include <cl/workloads/ClWorkloadUtils.hpp>

#include <backendsCommon/test/ActivationFixture.hpp>
#include <backendsCommon/test/LayerTests.hpp>

#include <arm_compute/core/CL/CLKernelLibrary.h>
#include <arm_compute/runtime/CL/CLScheduler.h>

#include <doctest/doctest.h>

#include <iostream>
#include <string>

TEST_SUITE("Compute_ArmComputeCl")
{

using namespace armnn;

using FactoryType = ClWorkloadFactory;

// ============================================================================
// UNIT tests

// Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ConstantLinearActivation, ClContextControlFixture, ConstantLinearActivationTest)

// Sigmoid Activation / Logistic
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleSigmoid, ClContextControlFixture, SimpleSigmoidTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleSigmoidUint8, ClContextControlFixture, SimpleSigmoidUint8Test)

// BoundedReLU Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReLu1, ClContextControlFixture, BoundedReLuUpperAndLowerBoundTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReLu6, ClContextControlFixture, BoundedReLuUpperBoundOnlyTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReLu1Uint8, ClContextControlFixture, BoundedReLuUint8UpperAndLowerBoundTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReLu6Uint8, ClContextControlFixture, BoundedReLuUint8UpperBoundOnlyTest)

// ReLU Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReLu, ClContextControlFixture, ReLuTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReLuUint8, ClContextControlFixture, ReLuUint8Test)

// SoftReLU Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SoftReLu, ClContextControlFixture, SoftReLuTest)

// LeakyReLU Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LeakyReLu, ClContextControlFixture, LeakyReLuTest)

// Abs Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Abs, ClContextControlFixture, AbsTest)

// Sqrt Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Sqrt, ClContextControlFixture, SqrtTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SqrtNN, ClContextControlFixture, SqrtNNTest)

// Square Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Square, ClContextControlFixture, SquareTest)

// Tanh Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Tanh, ClContextControlFixture, TanhTest)

// Elu Activation
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Elu, ClContextControlFixture, EluTest)

// Batch Mat Mul
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchMatMul2DSimpleFloat32,
                                 ClContextControlFixture,
                                 BatchMatMul2DSimpleTest<DataType::Float32>);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchMatMul3DSimpleFloat32,
                                 ClContextControlFixture,
                                 BatchMatMul3DSimpleTest<DataType::Float32>);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchMatMulNCHWSimpleFloat32,
                                 ClContextControlFixture,
                                 BatchMatMulNCHWSimpleTest<DataType::Float32>);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchMatMul3DBatchFloat32,
                                 ClContextControlFixture,
                                 BatchMatMul3DBatchTest<DataType::Float32>);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchMatMul3DBroadcastFloat32,
                                 ClContextControlFixture,
                                 BatchMatMul3DBroadcastTest<DataType::Float32>);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchMatMul3D2DBroadcastFloat32,
                                 ClContextControlFixture,
                                 BatchMatMul3D2DBroadcastTest<DataType::Float32>);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchMatMul2DTinyFloat32,
                                 ClContextControlFixture,
                                 BatchMatMul2DTinyTest<DataType::Float32>);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchMatMul2DTranspSimpleFloat32,
                                 ClContextControlFixture,
                                 BatchMatMul2DTranspSimpleTest<DataType::Float32>);

// Batch To Space
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcFloat321,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest1<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcFloat322,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest2<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcFloat323,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest3<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcFloat324,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest4<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcFloat325,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest5<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcFloat326,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest6<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcFloat327,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest7<DataType::Float32>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwFloat321,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest1<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwFloat322,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest2<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwFloat323,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest3<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwFloat324,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest4<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwFloat325,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest5<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwFloat326,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest6<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwFloat327,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest7<DataType::Float32>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcInt1,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest1<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcInt2,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest2<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcInt3,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest3<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcInt14,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest4<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcInt5,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest5<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcInt6,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest6<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcInt7,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest7<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwInt1,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest1<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwInt2,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest2<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwInt3,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest3<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwInt4,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest4<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwInt5,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest5<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwInt6,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest6<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwInt7,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest7<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcUint1,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest1<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcUint2,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest2<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcUint3,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest3<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcUint4,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest4<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcUint5,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest5<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcUint6,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest6<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNhwcUint7,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNhwcTest7<DataType::QAsymmU8>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwUint1,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest1<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwUint2,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest2<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwUint3,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest3<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwUint14,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest4<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwUint5,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest5<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwUint6,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest6<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchToSpaceNdNchwUint7,
                                 ClContextControlFixture,
                                 BatchToSpaceNdNchwTest7<DataType::QAsymmU8>)

// Fully Connected
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleFullyConnected,
                                 ClContextControlFixture,
                                 FullyConnectedFloat32Test,
                                 false,
                                 false)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleFullyConnectedWithBias,
                                 ClContextControlFixture,
                                 FullyConnectedFloat32Test,
                                 true,
                                 false)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleFullyConnectedWithTranspose,
                                 ClContextControlFixture,
                                 FullyConnectedFloat32Test,
                                 false,
                                 true)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(FullyConnectedUint8,
                                 ClContextControlFixture,
                                 FullyConnectedTest<DataType::QAsymmU8>,
                                 false,
                                 true)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(FullyConnectedBiasedUint8,
                                 ClContextControlFixture,
                                 FullyConnectedTest<DataType::QAsymmU8>,
                                 true,
                                 true)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(FullyConnectedLarge,
                                 ClContextControlFixture,
                                 FullyConnectedLargeTest,
                                 false)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(FullyConnectedLargeTransposed,
                                 ClContextControlFixture,
                                 FullyConnectedLargeTest,
                                 true)

// Convolution
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvolution1d,
                                 ClContextControlFixture,
                                 Convolution1dTest,
                                 true)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvolution2d,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x5Test,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvolution2dNhwc,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x5Test,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvolution2d3x3Uint8,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x3Uint8Test,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvolution2d3x3Uint8Nhwc,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x3Uint8Test,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedConvolution2d,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x5Test,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedConvolution2dNhwc,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x5Test,
                                 false,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedConvolution2dStride2x2Nhwc,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x3Stride2x2Test,
                                 false,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedConvolution2dSquare,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x3Test,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvolution2dAsymmetricPadding,
                                 ClContextControlFixture,
                                 Convolution2dAsymmetricPaddingTest,
                                 DataLayout::NCHW)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedConvolution2dSquareNhwc,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x3Test,
                                 false,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvolution2dAsymmetricPaddingNhwc,
                                 ClContextControlFixture,
                                 Convolution2dAsymmetricPaddingTest,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvolution2dSquareNhwc,
                                 ClContextControlFixture,
                                 SimpleConvolution2d3x3NhwcTest,
                                 false)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d3x3Dilation3x3,
                                 ClContextControlFixture,
                                 Convolution2d3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d3x3Dilation3x3Nhwc,
                                 ClContextControlFixture,
                                 Convolution2d3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                                 false,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d3x3Dilation3x3Uint8,
                                 ClContextControlFixture,
                                 Convolution2d3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d3x3Dilation3x3NhwcUint8,
                                 ClContextControlFixture,
                                 Convolution2d3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                                 false,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d2x3x3Dilation3x3,
                                 ClContextControlFixture,
                                 Convolution2d2x3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d2x3x3Dilation3x3Nhwc,
                                 ClContextControlFixture,
                                 Convolution2d2x3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                                 false,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d2x3x3Dilation3x3Uint8,
                                 ClContextControlFixture,
                                 Convolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d2x3x3Dilation3x3NhwcUint8,
                                 ClContextControlFixture,
                                 Convolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                                 false,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3,
        ClContextControlFixture,
        Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::Float32, DataType::Float32>,
        false,
        DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Nhwc,
        ClContextControlFixture,
        Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::Float32, DataType::Float32>,
        false,
        DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Uint8,
        ClContextControlFixture,
        Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QAsymmU8, DataType::Signed32>,
        false,
        DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3NhwcUint8,
        ClContextControlFixture,
        Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QAsymmU8, DataType::Signed32>,
        false,
        DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2dPerAxisQuantTestNchw,
                                 ClContextControlFixture,
                                 Convolution2dPerAxisQuantTest,
                                 DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Convolution2dPerAxisQuantTestNhwc,
                                 ClContextControlFixture,
                                 Convolution2dPerAxisQuantTest,
                                 DataLayout::NHWC);

// Convolution 3d - NDHWC
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution3d3x3x3Float32,
                              SimpleConvolution3d3x3x3Float32Test,
                              false,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution3d3x3x3Int8,
                              SimpleConvolution3d3x3x3Int8Test,
                              false,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution3d3x3x3Uint8,
                              SimpleConvolution3d3x3x3Uint8Test,
                              false,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3d2x2x2Strides3x5x5Float32,
                              Convolution3d2x2x2Strides3x5x5Float32Test,
                              false,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3d2x2x2Strides3x5x5TestInt8,
                              Convolution3d2x2x2Strides3x5x5Int8Test,
                              true,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3d2x2x2Strides3x5x5TestUint8,
                              Convolution3d2x2x2Strides3x5x5Uint8Test,
                              false,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3dPaddingSame3x3x3Float32,
                              Convolution3dPaddingSame3x3x3Float32Test,
                              false,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3dPaddingSame3x3x3TestInt8,
                              Convolution3dPaddingSame3x3x3Int8Test,
                              false,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3dPaddingSame3x3x3TestUint8,
                              Convolution3dPaddingSame3x3x3Uint8Test,
                              false,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3d2x2x2Stride3x3x3SmallTestFloat32,
                              Convolution3d2x2x2Stride3x3x3SmallFloat32Test,
                              false,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3d2x3x3TestFloat16,
                              Convolution3d2x3x3Float16Test,
                              true,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3d2x2x2SmallTestFloat16,
                              Convolution3d2x2x2SmallFloat16Test,
                              false,
                              DataLayout::NDHWC)

// Depthwise Convolution
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dDepthMul1,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthMul1Test,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthMul1Test,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dDepthMul1Uint8,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthMul1Uint8Test,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Uint8,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthMul1Uint8Test,
                                 false,
                                 DataLayout::NCHW)

// NHWC Depthwise Convolution
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dDepthMul1Nhwc,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthMul1Test,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Nhwc,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthMul1Test,
                                 false,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dDepthMul1Uint8Nhwc,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthMul1Uint8Test,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Uint8Nhwc,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthMul1Uint8Test,
                                 false,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleDepthwiseConvolution2d3x3Dilation3x3Nhwc,
                                 ClContextControlFixture,
                                 SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest)


ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dDepthNhwc,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthNhwcTest,
                                 false)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dAsymmetric,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dAsymmetricTest,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedDepthwiseConvolution2dAsymmetric,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dAsymmetricTest,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dAsymmetricNhwc,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dAsymmetricTest,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedDepthwiseConvolution2dAsymmetricNhwc,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dAsymmetricTest,
                                 false,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dDepthMul64,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dDepthMul64Test);

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dPerAxisQuantTestNchw,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dPerAxisQuantTest,
                                 DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DepthwiseConvolution2dPerAxisQuantTestNhwc,
                                 ClContextControlFixture,
                                 DepthwiseConvolution2dPerAxisQuantTest,
                                 DataLayout::NHWC);

// Splitter
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleSplitterFloat32, ClContextControlFixture, SplitterFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleSplitterUint8, ClContextControlFixture, SplitterUint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(CopyViaSplitterFloat32, ClContextControlFixture, CopyViaSplitterFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(CopyViaSplitterUint8, ClContextControlFixture, CopyViaSplitterUint8Test)

// Concat
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConcat, ClContextControlFixture, ConcatTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ConcatUint8, ClContextControlFixture, ConcatUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ConcatUint8DifferentInputOutputQParam,
                                 ClContextControlFixture,
                                 ConcatDifferentInputOutputQParamTest<DataType::QAsymmU8>,
                                 false)

// Normalization
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleNormalizationAcross, ClContextControlFixture, SimpleNormalizationAcrossTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleNormalizationWithin, ClContextControlFixture, SimpleNormalizationWithinTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleNormalizationAcrossNhwc,
                                 ClContextControlFixture,
                                 SimpleNormalizationAcrossNhwcTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AcrossChannelNormalization,
                                 ClContextControlFixture,
                                 AcrossChannelNormalizationTest)

// Pooling2d
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling2dSize3x3Stride2x4,
                                 ClContextControlFixture,
                                 SimpleMaxPooling2dSize3x3Stride2x4Test,
                                 true)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling2dSize3x3Stride2x4Uint8,
                                 ClContextControlFixture,
                                 SimpleMaxPooling2dSize3x3Stride2x4Uint8Test,
                                 true)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleMaxPooling2d,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleMaxPooling2dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleMaxPooling2dUint8,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleMaxPooling2dUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingMaxPooling2dSize3,
                                 ClContextControlFixture,
                                 IgnorePaddingMaxPooling2dSize3Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingMaxPooling2dSize3Uint8,
                                 ClContextControlFixture,
                                 IgnorePaddingMaxPooling2dSize3Uint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleAveragePooling2d,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleAveragePooling2dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleAveragePooling2dUint8,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleAveragePooling2dUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleAveragePooling2dNoPadding,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleAveragePooling2dNoPaddingTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleAveragePooling2dNoPaddingUint8,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleAveragePooling2dNoPaddingUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingAveragePooling2dSize3,
                                 ClContextControlFixture,
                                 IgnorePaddingAveragePooling2dSize3Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingAveragePooling2dSize3Uint8,
                                 ClContextControlFixture,
                                 IgnorePaddingAveragePooling2dSize3Uint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleL2Pooling2d,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleL2Pooling2dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_IgnorePaddingSimpleL2Pooling2dUint8,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleL2Pooling2dUint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingL2Pooling2dSize3,
                                 ClContextControlFixture,
                                 IgnorePaddingL2Pooling2dSize3Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_IgnorePaddingL2Pooling2dSize3Uint8,
                                 ClContextControlFixture,
                                 IgnorePaddingL2Pooling2dSize3Uint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling2d,
                                 ClContextControlFixture,
                                 SimpleMaxPooling2dTest,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling2dNhwc,
                                 ClContextControlFixture,
                                 SimpleMaxPooling2dTest,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling2dUint8,
                                 ClContextControlFixture,
                                 SimpleMaxPooling2dUint8Test,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling2dUint8Nhwc,
                                 ClContextControlFixture,
                                 SimpleMaxPooling2dUint8Test,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleAveragePooling2d,
                                 ClContextControlFixture,
                                 SimpleAveragePooling2dTest,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleAveragePooling2dNhwc,
                                 ClContextControlFixture,
                                 SimpleAveragePooling2dTest,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleAveragePooling2dUint8,
                                 ClContextControlFixture,
                                 SimpleAveragePooling2dUint8Test,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleAveragePooling2dUint8Nhwc,
                                 ClContextControlFixture,
                                 SimpleAveragePooling2dUint8Test,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingAveragePooling2dSize3x2Stride2x2,
                                 ClContextControlFixture,
                                 IgnorePaddingAveragePooling2dSize3x2Stride2x2Test,
                                 false)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingAveragePooling2dSize3x2Stride2x2NoPadding,
                                 ClContextControlFixture,
                                 IgnorePaddingAveragePooling2dSize3x2Stride2x2Test,
                                 true)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LargeTensorsAveragePooling2d,
                                 ClContextControlFixture,
                                 LargeTensorsAveragePooling2dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LargeTensorsAveragePooling2dUint8,
                                 ClContextControlFixture,
                                 LargeTensorsAveragePooling2dUint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleL2Pooling2d,
                                 ClContextControlFixture,
                                 SimpleL2Pooling2dTest,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleL2Pooling2dNhwc,
                                 ClContextControlFixture,
                                 SimpleL2Pooling2dTest,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_SimpleL2Pooling2dUint8,
                                 ClContextControlFixture,
                                 SimpleL2Pooling2dUint8Test,
                                 DataLayout::NCHW)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Pooling2dSize3Stride1, ClContextControlFixture, L2Pooling2dSize3Stride1Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_L2Pooling2dSize3Stride1Uint8,
                                 ClContextControlFixture,
                                 L2Pooling2dSize3Stride1Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Pooling2dSize3Stride3,
                                 ClContextControlFixture,
                                 L2Pooling2dSize3Stride3Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_L2Pooling2dSize3Stride3Uint8,
                                 ClContextControlFixture,
                                 L2Pooling2dSize3Stride3Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Pooling2dSize3Stride4,
                                 ClContextControlFixture,
                                 L2Pooling2dSize3Stride4Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_L2Pooling2dSize3Stride4Uint8,
                                 ClContextControlFixture,
                                 L2Pooling2dSize3Stride4Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Pooling2dSize7,
                                 ClContextControlFixture,
                                 L2Pooling2dSize7Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_L2Pooling2dSize7Uint8,
                                 ClContextControlFixture,
                                 L2Pooling2dSize7Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Pooling2dSize9, ClContextControlFixture, L2Pooling2dSize9Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_L2Pooling2dSize9Uint8, ClContextControlFixture, L2Pooling2dSize9Uint8Test)

// Pooling3d
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling3dSize2x2x2Stride1x1x1,
                                 ClContextControlFixture,
                                 SimpleMaxPooling3dSize2x2x2Stride1x1x1Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling3dSize2x2x2Stride1x1x1Uint8,
                                 ClContextControlFixture,
                                 SimpleMaxPooling3dSize2x2x2Stride1x1x1Uint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling3d,
                                 ClContextControlFixture,
                                 SimpleMaxPooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMaxPooling3dUint8,
                                 ClContextControlFixture,
                                 SimpleMaxPooling3dUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleMaxPooling3d,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleMaxPooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleMaxPooling3dUint8,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleMaxPooling3dUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleAveragePooling3d,
                                 ClContextControlFixture,
                                 SimpleAveragePooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleAveragePooling3dUint8,
                                 ClContextControlFixture,
                                 SimpleAveragePooling3dUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LargeTensorsAveragePooling3d,
                                 ClContextControlFixture,
                                 LargeTensorsAveragePooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LargeTensorsAveragePooling3dUint8,
                                 ClContextControlFixture,
                                 LargeTensorsAveragePooling3dUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleAveragePooling3d,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleAveragePooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleL2Pooling3d,
                                 ClContextControlFixture,
                                 SimpleL2Pooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(IgnorePaddingSimpleL2Pooling3d,
                                 ClContextControlFixture,
                                 IgnorePaddingSimpleL2Pooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AsymmetricNonSquareMaxPooling3d,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareMaxPooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AsymmetricNonSquareMaxPooling3dUint8,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareMaxPooling3dUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AsymmetricNonSquareAveragePooling3d,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareAveragePooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AsymmetricNonSquareAveragePooling3dUint8,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareAveragePooling3dUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AsymmetricNonSquareL2Pooling3d,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareL2Pooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPool,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolUint8,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareMaxPooling3dWithPaddingOnlyPoolUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolUint8,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTED_AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolUint8,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareAveragePooling3dWithPaddingOnlyPoolUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTEDAsymmetricNonSquareL2Pooling3dWithPaddingOnlyPool,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UNSUPPORTEDAsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolUint8,
                                 ClContextControlFixture,
                                 AsymmetricNonSquareL2Pooling3dWithPaddingOnlyPoolUint8Test,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleAveragePooling3d,
                                 ClContextControlFixture,
                                 SimpleAveragePooling3dTest,
                                 DataLayout::NDHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleAveragePooling3dUint8,
                                 ClContextControlFixture,
                                 SimpleAveragePooling3dUint8Test,
                                 DataLayout::NDHWC)

// Add
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleAdd, ClContextControlFixture, AdditionTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Add5d, ClContextControlFixture, Addition5dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AddBroadcast1Element, ClContextControlFixture, AdditionBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AddBroadcast, ClContextControlFixture, AdditionBroadcastTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AdditionUint8, ClContextControlFixture, AdditionUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AddBroadcastUint8, ClContextControlFixture, AdditionBroadcastUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AddBroadcast1ElementUint8,
                                 ClContextControlFixture,
                                 AdditionBroadcast1ElementUint8Test)

// Sub
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleSub, ClContextControlFixture, SubtractionTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SubBroadcast1Element, ClContextControlFixture, SubtractionBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SubBroadcast, ClContextControlFixture, SubtractionBroadcastTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SubtractionUint8, ClContextControlFixture, SubtractionUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SubBroadcastUint8, ClContextControlFixture, SubtractionBroadcastUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SubBroadcast1ElementUint8,
                                 ClContextControlFixture,
                                 SubtractionBroadcast1ElementUint8Test)

// Div
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleDivision, ClContextControlFixture, DivisionTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DivisionByZero, ClContextControlFixture, DivisionByZeroTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DivisionBroadcast1Element, ClContextControlFixture, DivisionBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(DivisionBroadcast1DVector, ClContextControlFixture, DivisionBroadcast1DVectorTest)
// NOTE: quantized division is not supported by CL and not required by the
//       android NN api

// Mul
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleMultiplication, ClContextControlFixture, MultiplicationTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MultiplicationBroadcast1Element,
                                 ClContextControlFixture,
                                 MultiplicationBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MultiplicationBroadcast1DVector,
                                 ClContextControlFixture,
                                 MultiplicationBroadcast1DVectorTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MultiplicationUint8, ClContextControlFixture, MultiplicationUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MultiplicationBroadcast1ElementUint8,
                                 ClContextControlFixture,
                                 MultiplicationBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MultiplicationBroadcast1DVectorUint8,
                                 ClContextControlFixture,
                                 MultiplicationBroadcast1DVectorUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Multiplication5d, ClContextControlFixture, Multiplication5dTest)

// Batch Norm
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchNormFloat32, ClContextControlFixture, BatchNormFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(BatchNormFloat32Nhwc, ClContextControlFixture, BatchNormFloat32NhwcTest)

// Rank
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize1Float16, ClContextControlFixture, RankDimSize1Test<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize1Float32, ClContextControlFixture, RankDimSize1Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize1QAsymmU8, ClContextControlFixture, RankDimSize1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize1Signed32, ClContextControlFixture, RankDimSize1Test<DataType::Signed32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize1QSymmS16, ClContextControlFixture, RankDimSize1Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize1QAsymmS8, ClContextControlFixture, RankDimSize1Test<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize2Float16, ClContextControlFixture, RankDimSize2Test<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize2Float32, ClContextControlFixture, RankDimSize2Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize2QAsymmU8, ClContextControlFixture, RankDimSize2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize2Signed32, ClContextControlFixture, RankDimSize2Test<DataType::Signed32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize2QSymmS16, ClContextControlFixture, RankDimSize2Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize2QAsymmS8, ClContextControlFixture, RankDimSize2Test<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize3Float16, ClContextControlFixture, RankDimSize3Test<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize3Float32, ClContextControlFixture, RankDimSize3Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize3QAsymmU8, ClContextControlFixture, RankDimSize3Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize3Signed32, ClContextControlFixture, RankDimSize3Test<DataType::Signed32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize3QSymmS16, ClContextControlFixture, RankDimSize3Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize3QAsymmS8, ClContextControlFixture, RankDimSize3Test<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize4Float16, ClContextControlFixture, RankDimSize4Test<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize4Float32, ClContextControlFixture, RankDimSize4Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize4QAsymmU8, ClContextControlFixture, RankDimSize4Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize4Signed32, ClContextControlFixture, RankDimSize4Test<DataType::Signed32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize4QSymmS16, ClContextControlFixture, RankDimSize4Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RankDimSize4QAsymmS8, ClContextControlFixture, RankDimSize4Test<DataType::QAsymmS8>)

// InstanceNormalization
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(InstanceNormFloat32Nchw,
                                 ClContextControlFixture,
                                 InstanceNormFloat32Test,
                                 DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(InstanceNormFloat16Nchw,
                                 ClContextControlFixture,
                                 InstanceNormFloat16Test,
                                 DataLayout::NCHW);

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(InstanceNormFloat32Nhwc,
                                 ClContextControlFixture,
                                 InstanceNormFloat32Test,
                                 DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(InstanceNormFloat16Nhwc,
                                 ClContextControlFixture,
                                 InstanceNormFloat16Test,
                                 DataLayout::NHWC);

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(InstanceNormFloat32Nchw2,
                                 ClContextControlFixture,
                                 InstanceNormFloat32Test2,
                                 DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(InstanceNormFloat16Nchw2,
                                 ClContextControlFixture,
                                 InstanceNormFloat16Test2,
                                 DataLayout::NCHW);

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(InstanceNormFloat32Nhwc2,
                                 ClContextControlFixture,
                                 InstanceNormFloat32Test2,
                                 DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(InstanceNormFloat16Nhwc2,
                                 ClContextControlFixture,
                                 InstanceNormFloat16Test2,
                                 DataLayout::NHWC);

// L2 Normalization
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Normalization1d, ClContextControlFixture, L2Normalization1dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Normalization2d, ClContextControlFixture, L2Normalization2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Normalization3d, ClContextControlFixture, L2Normalization3dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Normalization4d, ClContextControlFixture, L2Normalization4dTest, DataLayout::NCHW)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Normalization1dNhwc,
                                 ClContextControlFixture,
                                 L2Normalization1dTest,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Normalization2dNhwc,
                                 ClContextControlFixture,
                                 L2Normalization2dTest,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Normalization3dNhwc,
                                 ClContextControlFixture,
                                 L2Normalization3dTest,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Normalization4dNhwc,
                                 ClContextControlFixture,
                                 L2Normalization4dTest,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2Normalization2dShape, ClContextControlFixture, L2Normalization2dShapeTest);

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2NormalizationDefaultEpsilon,
                                 ClContextControlFixture,
                                 L2NormalizationDefaultEpsilonTest,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(L2NormalizationNonDefaultEpsilon,
                                 ClContextControlFixture,
                                 L2NormalizationNonDefaultEpsilonTest,
                                 DataLayout::NCHW)

// Constant
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Constant, ClContextControlFixture, ConstantTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ConstantUint8,
                                 ClContextControlFixture,
                                 ConstantUint8SimpleQuantizationScaleNoOffsetTest)

// Concat
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat1d, ClContextControlFixture, Concat1dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat1dUint8, ClContextControlFixture, Concat1dUint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat2dDim0, ClContextControlFixture, Concat2dDim0Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat2dDim0Uint8, ClContextControlFixture, Concat2dDim0Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat2dDim1, ClContextControlFixture, Concat2dDim1Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat2dDim1Uint8, ClContextControlFixture, Concat2dDim1Uint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat2dDim0DiffInputDims,
                                 ClContextControlFixture,
                                 Concat2dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat2dDim0DiffInputDimsUint8,
                                 ClContextControlFixture,
                                 Concat2dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat2dDim1DiffInputDims,
                                 ClContextControlFixture,
                                 Concat2dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat2dDim1DiffInputDimsUint8,
                                 ClContextControlFixture,
                                 Concat2dDim1DiffInputDimsUint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim0, ClContextControlFixture, Concat3dDim0Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim0Uint8, ClContextControlFixture, Concat3dDim0Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim1, ClContextControlFixture, Concat3dDim1Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim1Uint8, ClContextControlFixture, Concat3dDim1Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim2, ClContextControlFixture, Concat3dDim2Test, false)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim2Uint8, ClContextControlFixture, Concat3dDim2Uint8Test, false)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim0DiffInputDims, ClContextControlFixture, Concat3dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim0DiffInputDimsUint8,
                                 ClContextControlFixture,
                                 Concat3dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim1DiffInputDims,
                                 ClContextControlFixture,
                                 Concat3dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim1DiffInputDimsUint8,
                                 ClContextControlFixture,
                                 Concat3dDim1DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim2DiffInputDims,
                                 ClContextControlFixture,
                                 Concat3dDim2DiffInputDimsTest,
                                 false)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat3dDim2DiffInputDimsUint8,
                                 ClContextControlFixture,
                                 Concat3dDim2DiffInputDimsUint8Test,
                                 false)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDim0, ClContextControlFixture, Concat4dDim0Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDim1, ClContextControlFixture, Concat4dDim1Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDim3, ClContextControlFixture, Concat4dDim3Test, false)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDim0Uint8, ClContextControlFixture, Concat4dDim0Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDim1Uint8, ClContextControlFixture, Concat4dDim1Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDim3Uint8, ClContextControlFixture, Concat4dDim3Uint8Test, false)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDiffShapeDim0, ClContextControlFixture, Concat4dDiffShapeDim0Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDiffShapeDim1, ClContextControlFixture, Concat4dDiffShapeDim1Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDiffShapeDim3, ClContextControlFixture, Concat4dDiffShapeDim3Test, false)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDiffShapeDim0Uint8, ClContextControlFixture, Concat4dDiffShapeDim0Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDiffShapeDim1Uint8, ClContextControlFixture, Concat4dDiffShapeDim1Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Concat4dDiffShapeDim3Uint8,
                                 ClContextControlFixture,
                                 Concat4dDiffShapeDim3Uint8Test,
                                 false)

// DepthToSpace
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwFloat32_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::Float32>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwFloat32_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::Float32>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwFloat32_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::Float32>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwFloat32_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::Float32>, DataLayout::NCHW);

ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwFloat16_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::Float16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwFloat16_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::Float16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwFloat16_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::Float16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwFloat16_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::Float16>, DataLayout::NCHW);

ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwInt8_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::QAsymmS8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwInt8_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::QAsymmS8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwInt8_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::QAsymmS8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwInt8_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::QAsymmS8>, DataLayout::NCHW);

ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwUint8_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::QAsymmU8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwUint8_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::QAsymmU8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwUint8_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::QAsymmU8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwUint8_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::QAsymmU8>, DataLayout::NCHW);

ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwInt16_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::QSymmS16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwInt16_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::QSymmS16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwInt16_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::QSymmS16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNchwInt16_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::QSymmS16>, DataLayout::NCHW);

ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcFloat32_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::Float32>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcFloat32_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::Float32>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcFloat32_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::Float32>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcFloat32_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::Float32>, DataLayout::NHWC);

ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcFloat16_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::Float16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcFloat16_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::Float16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcFloat16_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::Float16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcFloat16_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::Float16>, DataLayout::NHWC);

ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcInt8_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::QAsymmS8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcInt8_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::QAsymmS8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcInt8_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::QAsymmS8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcInt8_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::QAsymmS8>, DataLayout::NHWC);

ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcUint8_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::QAsymmU8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcUint8_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::QAsymmU8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcUint8_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::QAsymmU8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcUint8_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::QAsymmU8>, DataLayout::NHWC);

ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcInt16_1,
    ClContextControlFixture, DepthToSpaceTest1<DataType::QSymmS16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcInt16_2,
    ClContextControlFixture, DepthToSpaceTest2<DataType::QSymmS16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcInt16_3,
    ClContextControlFixture, DepthToSpaceTest3<DataType::QSymmS16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_FIXTURE(DepthToSpaceNhwcInt16_4,
    ClContextControlFixture, DepthToSpaceTest4<DataType::QSymmS16>, DataLayout::NHWC);

// Fill
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleFill, ClContextControlFixture, SimpleFillTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleFillF16, ClContextControlFixture, SimpleFillTest<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleFillS32, ClContextControlFixture, SimpleFillTest<DataType::Signed32>)

// FloorPreluUint8
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleFloor, ClContextControlFixture, SimpleFloorTest<DataType::Float32>)

// Gather
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Gather1dParamsFloat32, ClContextControlFixture, Gather1dParamsFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Gather1dParamsUint8, ClContextControlFixture, Gather1dParamsUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherMultiDimParamsFloat32, ClContextControlFixture, GatherMultiDimParamsFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherMultiDimParamsUint8, ClContextControlFixture, GatherMultiDimParamsUint8Test)

// GatherNd
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherNd2dFloat32, ClContextControlFixture, SimpleGatherNd2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherNd3dFloat32, ClContextControlFixture, SimpleGatherNd3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherNd4dFloat32, ClContextControlFixture, SimpleGatherNd4dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherNd2dInt8, ClContextControlFixture, SimpleGatherNd2dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherNd3dInt8, ClContextControlFixture, SimpleGatherNd3dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherNd4dInt8, ClContextControlFixture, SimpleGatherNd4dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherNd2dInt32, ClContextControlFixture, SimpleGatherNd2dTest<DataType::Signed32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherNd3dInt32, ClContextControlFixture, SimpleGatherNd3dTest<DataType::Signed32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GatherNd4dInt32, ClContextControlFixture, SimpleGatherNd4dTest<DataType::Signed32>)

// Reshape
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleReshapeFloat32, ClContextControlFixture, SimpleReshapeTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleReshapeInt8, ClContextControlFixture, SimpleReshapeTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleReshapeUint8, ClContextControlFixture, SimpleReshapeTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Reshape5d, ClContextControlFixture, Reshape5dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReshapeBoolean, ClContextControlFixture, ReshapeBooleanTest)

// Pad - Constant
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PadFloat322d, ClContextControlFixture, PadFloat322dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PadFloat322dCustomPadding, ClContextControlFixture, PadFloat322dCustomPaddingTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PadFloat323d, ClContextControlFixture, PadFloat323dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PadFloat324d, ClContextControlFixture, PadFloat324dTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PadUint82d, ClContextControlFixture, PadUint82dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PadUint82dCustomPadding, ClContextControlFixture, PadUint82dCustomPaddingTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PadUint83d, ClContextControlFixture, PadUint83dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PadUint84d, ClContextControlFixture, PadUint84dTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Pad2dQSymm16,
    ClContextControlFixture, Pad2dTestCommon<DataType::QSymmS16>, 2.0f, 0, 0.0f)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Pad2dQSymm16CustomPadding,
    ClContextControlFixture, Pad2dTestCommon<DataType::QSymmS16>, 2.0f, 0, 1.0f)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Pad3dQSymm16, ClContextControlFixture, Pad3dTestCommon<DataType::QSymmS16>, 2.0f, 0)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Pad4dQSymm16, ClContextControlFixture, Pad4dTestCommon<DataType::QSymmS16>, 2.0f, 0)

// Pad - Symmetric & Reflect
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric2dFloat32, PadSymmetric2dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect2dFloat32, PadReflect2dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric3dFloat32, PadSymmetric3dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect3dFloat32, PadReflect3dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric3dUint8, PadSymmetric3dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect3dUint8, PadReflect3dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric3dInt8, PadSymmetric3dInt8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect3dInt8, PadReflect3dInt8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetricFloat16, PadSymmetricFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflectFloat16, PadReflectFloat16Test)

// PReLU
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PreluFloat32, ClContextControlFixture, PreluTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PreluUint8, ClContextControlFixture,  PreluTest<DataType::QAsymmU8>)

// Permute
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimplePermuteFloat32, ClContextControlFixture, SimplePermuteTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    PermuteFloat32ValueSet1Test, ClContextControlFixture, PermuteValueSet1Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    PermuteFloat32ValueSet2Test, ClContextControlFixture, PermuteValueSet2Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    PermuteFloat32ValueSet3Test, ClContextControlFixture, PermuteValueSet3Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SimplePermuteQASymmS8, ClContextControlFixture, SimplePermuteTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    PermuteQASymmS8ValueSet1Test, ClContextControlFixture, PermuteValueSet1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    PermuteQASymmS8ValueSet2Test, ClContextControlFixture, PermuteValueSet2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    PermuteQASymmS8ValueSet3Test, ClContextControlFixture, PermuteValueSet3Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SimplePermuteQASymm8, ClContextControlFixture, SimplePermuteTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    PermuteQASymm8ValueSet1Test, ClContextControlFixture, PermuteValueSet1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    PermuteQASymm8ValueSet2Test, ClContextControlFixture, PermuteValueSet2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    PermuteQASymm8ValueSet3Test, ClContextControlFixture, PermuteValueSet3Test<DataType::QAsymmU8>)

// Lstm
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LstmLayerFloat32WithCifgWithPeepholeNoProjection, ClContextControlFixture,
                              LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LstmLayerFloat32NoCifgNoPeepholeNoProjection, ClContextControlFixture,
                              LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LstmLayerFloat32NoCifgWithPeepholeWithProjection, ClContextControlFixture,
                              LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNorm,
                                 ClContextControlFixture,
                                 LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNormTest)

// QLstm
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(QLstm, ClContextControlFixture, QLstmTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(QLstm1, ClContextControlFixture, QLstmTest1)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(QLstm2, ClContextControlFixture, QLstmTest2)

// QuantizedLstm
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(QuantizedLstm, ClContextControlFixture, QuantizedLstmTest)

// Unidirectional Sequence Lstm
ARMNN_AUTO_TEST_CASE_WITH_THF(UnidirectionalSequenceLstmLayerFloat32TimeMajorSingleBatch,
                              UnidirectionalSequenceLstmLayerFloat32TimeMajorSingleBatchTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnidirectionalSequenceLstmLayerFloat32BatchMajorSingleBatch,
                              UnidirectionalSequenceLstmLayerFloat32BatchMajorSingleBatchTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnidirectionalSequenceLstmLayerFloat32,
                              UnidirectionalSequenceLstmLayerFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnidirectionalSequenceLstmLayerFloat32TimeMajor,
                              UnidirectionalSequenceLstmLayerFloat32TimeMajorTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjection,
                              UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionWithLayerNorm,
                              UnidirectionalSequenceLstmLayerNoCifgWithPeepholeWithProjectionWithLayerNormTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnidirectionalSequenceLstmWithCifgWithPeepholeNoProjection,
                              UnidirectionalSequenceLstmWithCifgWithPeepholeNoProjectionTest)

// Convert from Float16 to Float32
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvertFp16ToFp32, ClContextControlFixture, SimpleConvertFp16ToFp32Test)
// Convert from Float32 to Float16
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleConvertFp32ToFp16, ClContextControlFixture, SimpleConvertFp32ToFp16Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AdditionAfterMaxPool, ClContextControlFixture, AdditionAfterMaxPoolTest)

//Max
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MaximumSimple, ClContextControlFixture, MaximumSimpleTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MaximumBroadcast1Element, ClContextControlFixture, MaximumBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MaximumBroadcast1DVector, ClContextControlFixture, MaximumBroadcast1DVectorTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MaximumUint8, ClContextControlFixture, MaximumUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MaximumBroadcast1ElementUint8, ClContextControlFixture, MaximumBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MaximumBroadcast1DVectorUint8, ClContextControlFixture, MaximumBroadcast1DVectorUint8Test)

// Mean
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanSimpleFloat32, ClContextControlFixture, MeanSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanSimpleAxisFloat32, ClContextControlFixture, MeanSimpleAxisTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanKeepDimsFloat32, ClContextControlFixture, MeanKeepDimsTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MeanMultipleDimsFloat32, ClContextControlFixture, MeanMultipleDimsTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanVts1Float32, ClContextControlFixture, MeanVts1Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanVts2Float32, ClContextControlFixture, MeanVts2Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanVts3Float32, ClContextControlFixture, MeanVts3Test<DataType::Float32>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MeanSimpleQuantisedAsymmS8, ClContextControlFixture, MeanSimpleTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MeanSimpleAxisQuantisedAsymmS8, ClContextControlFixture, MeanSimpleAxisTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MeanKeepDimsQuantisedAsymmS8, ClContextControlFixture, MeanKeepDimsTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MeanMultipleDimsQuantisedAsymmS8, ClContextControlFixture, MeanMultipleDimsTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanVts1QuantisedAsymmS8, ClContextControlFixture, MeanVts1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanVts2QuantisedAsymmS8, ClContextControlFixture, MeanVts2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanVts3QuantisedAsymmS8, ClContextControlFixture, MeanVts3Test<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MeanSimpleQuantisedAsymm8, ClContextControlFixture, MeanSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MeanSimpleAxisQuantisedAsymm8, ClContextControlFixture, MeanSimpleAxisTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MeanKeepDimsQuantisedAsymm8, ClContextControlFixture, MeanKeepDimsTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MeanMultipleDimsQuantisedAsymm8, ClContextControlFixture, MeanMultipleDimsTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanVts1QuantisedAsymm8, ClContextControlFixture, MeanVts1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanVts2QuantisedAsymm8, ClContextControlFixture, MeanVts2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MeanVts3QuantisedAsymm8, ClContextControlFixture, MeanVts3Test<DataType::QAsymmU8>)

// Minimum
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MinimumBroadcast1Element1, ClContextControlFixture, MinimumBroadcast1ElementTest1)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MinimumBroadcast1Element2, ClContextControlFixture, MinimumBroadcast1ElementTest2)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    MinimumBroadcast1DVectorUint8, ClContextControlFixture, MinimumBroadcast1DVectorUint8Test)

// Equal
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(EqualSimple, ClContextControlFixture, EqualSimpleTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(EqualBroadcast1Element, ClContextControlFixture, EqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(EqualBroadcast1dVector, ClContextControlFixture, EqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(EqualSimpleFloat16, ClContextControlFixture, EqualSimpleFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    EqualBroadcast1ElementFloat16, ClContextControlFixture, EqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    EqualBroadcast1dVectorFloat16, ClContextControlFixture, EqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(EqualSimpleUint8,  ClContextControlFixture, EqualSimpleUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(EqualBroadcast1ElementUint8, ClContextControlFixture, EqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(EqualBroadcast1dVectorUint8, ClContextControlFixture, EqualBroadcast1dVectorUint8Test)

// Greater
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GreaterSimple, ClContextControlFixture, GreaterSimpleTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GreaterBroadcast1Element, ClContextControlFixture, GreaterBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GreaterBroadcast1dVector, ClContextControlFixture, GreaterBroadcast1dVectorTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GreaterSimpleFloat16, ClContextControlFixture, GreaterSimpleFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterBroadcast1ElementFloat16, ClContextControlFixture, GreaterBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterBroadcast1dVectorFloat16, ClContextControlFixture, GreaterBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GreaterSimpleUint8, ClContextControlFixture, GreaterSimpleUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterBroadcast1ElementUint8, ClContextControlFixture, GreaterBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterBroadcast1dVectorUint8, ClContextControlFixture, GreaterBroadcast1dVectorUint8Test)

// GreaterOrEqual
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GreaterOrEqualSimple, ClContextControlFixture, GreaterOrEqualSimpleTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterOrEqualBroadcast1Element, ClContextControlFixture, GreaterOrEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterOrEqualBroadcast1dVector, ClContextControlFixture, GreaterOrEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterOrEqualSimpleFloat16, ClContextControlFixture, GreaterOrEqualSimpleFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterOrEqualBroadcast1ElementFloat16, ClContextControlFixture, GreaterOrEqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterOrEqualBroadcast1dVectorFloat16, ClContextControlFixture, GreaterOrEqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(GreaterOrEqualSimpleUint8, ClContextControlFixture, GreaterOrEqualSimpleUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterOrEqualBroadcast1ElementUint8, ClContextControlFixture, GreaterOrEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    GreaterOrEqualBroadcast1dVectorUint8, ClContextControlFixture, GreaterOrEqualBroadcast1dVectorUint8Test)

// Less
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessSimple, ClContextControlFixture, LessSimpleTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessBroadcast1Element, ClContextControlFixture, LessBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessBroadcast1dVector, ClContextControlFixture, LessBroadcast1dVectorTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessSimpleFloat16, ClContextControlFixture, LessSimpleFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    LessBroadcast1ElementFloat16, ClContextControlFixture, LessBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    LessBroadcast1dVectorFloat16, ClContextControlFixture, LessBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessSimpleUint8, ClContextControlFixture, LessSimpleUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessBroadcast1ElementUint8, ClContextControlFixture, LessBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessBroadcast1dVectorUint8, ClContextControlFixture, LessBroadcast1dVectorUint8Test)

// LessOrEqual
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessOrEqualSimple, ClContextControlFixture, LessOrEqualSimpleTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    LessOrEqualBroadcast1Element, ClContextControlFixture, LessOrEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    LessOrEqualBroadcast1dVector, ClContextControlFixture, LessOrEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessOrEqualSimpleFloat16, ClContextControlFixture, LessOrEqualSimpleFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    LessOrEqualBroadcast1ElementFloat16, ClContextControlFixture, LessOrEqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    LessOrEqualBroadcast1dVectorFloat16, ClContextControlFixture, LessOrEqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LessOrEqualSimpleUint8, ClContextControlFixture, LessOrEqualSimpleUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    LessOrEqualBroadcast1ElementUint8, ClContextControlFixture, LessOrEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    LessOrEqualBroadcast1dVectorUint8, ClContextControlFixture, LessOrEqualBroadcast1dVectorUint8Test)

// NotEqual
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(NotEqualSimple, ClContextControlFixture, NotEqualSimpleTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(NotEqualBroadcast1Element, ClContextControlFixture, NotEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(NotEqualBroadcast1dVector, ClContextControlFixture, NotEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(NotEqualSimpleFloat16, ClContextControlFixture, NotEqualSimpleFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    NotEqualBroadcast1ElementFloat16, ClContextControlFixture, NotEqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    NotEqualBroadcast1dVectorFloat16, ClContextControlFixture, NotEqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(NotEqualSimpleUint8, ClContextControlFixture, NotEqualSimpleUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    NotEqualBroadcast1ElementUint8, ClContextControlFixture, NotEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    NotEqualBroadcast1dVectorUint8, ClContextControlFixture, NotEqualBroadcast1dVectorUint8Test)

// Softmax
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleSoftmaxBeta1, ClContextControlFixture, SimpleSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleSoftmaxBeta2, ClContextControlFixture, SimpleSoftmaxTest, 2.0f)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleSoftmaxBeta1Uint8, ClContextControlFixture, SimpleSoftmaxUint8Test, 1.0f)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleSoftmaxBeta2Uint8, ClContextControlFixture, SimpleSoftmaxUint8Test, 2.0f)

// LogSoftmax
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogSoftmaxFloat32_1, ClContextControlFixture, LogSoftmaxTest1<DataType::Float32>)

// Space To Batch Nd
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToBatchNdSimpleFloat32, ClContextControlFixture, SpaceToBatchNdSimpleFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdMultiChannelsFloat32, ClContextControlFixture, SpaceToBatchNdMultiChannelsFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdMultiBlockFloat32, ClContextControlFixture, SpaceToBatchNdMultiBlockFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdPaddingFloat32, ClContextControlFixture, SpaceToBatchNdPaddingFloat32Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToBatchNdSimpleUint8, ClContextControlFixture, SpaceToBatchNdSimpleUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdMultiChannelsUint8, ClContextControlFixture, SpaceToBatchNdMultiChannelsUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdMultiBlockUint8, ClContextControlFixture, SpaceToBatchNdMultiBlockUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdPaddingUint8, ClContextControlFixture, SpaceToBatchNdPaddingUint8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdSimpleNhwcFloat32, ClContextControlFixture, SpaceToBatchNdSimpleNhwcFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdMultiChannelsNhwcFloat32, ClContextControlFixture, SpaceToBatchNdMultiChannelsNhwcFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdMultiBlockNhwcFloat32, ClContextControlFixture, SpaceToBatchNdMultiBlockNhwcFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdPaddingNhwcFloat32, ClContextControlFixture, SpaceToBatchNdPaddingNhwcFloat32Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdSimpleNhwcUint8, ClContextControlFixture, SpaceToBatchNdSimpleNhwcUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdMultiChannelsNhwcUint8, ClContextControlFixture, SpaceToBatchNdMultiChannelsNhwcUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdMultiBlockNhwcUint8, ClContextControlFixture, SpaceToBatchNdMultiBlockNhwcUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SpaceToBatchNdPaddingNhwcUint8, ClContextControlFixture, SpaceToBatchNdPaddingNhwcUint8Test)

// Space To Depth
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToDepthNhwcAsymmQ8, ClContextControlFixture, SpaceToDepthNhwcAsymmQ8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToDepthNchwAsymmQ8, ClContextControlFixture, SpaceToDepthNchwAsymmQ8Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToDepthNhwx1Float32, ClContextControlFixture, SpaceToDepthNhwcFloat32Test1)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToDepthNchw1Float32, ClContextControlFixture, SpaceToDepthNchwFloat32Test1)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToDepthNhwc2Float32, ClContextControlFixture, SpaceToDepthNhwcFloat32Test2)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToDepthNchw2Float32, ClContextControlFixture, SpaceToDepthNchwFloat32Test2)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToDepthNhwcQSymm16, ClContextControlFixture, SpaceToDepthNhwcQSymm16Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SpaceToDepthNchwQSymm16, ClContextControlFixture, SpaceToDepthNchwQSymm16Test)

// Stack
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Stack0Axis, ClContextControlFixture, StackAxis0Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StackOutput4DAxis1, ClContextControlFixture, StackOutput4DAxis1Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StackOutput4DAxis2, ClContextControlFixture, StackOutput4DAxis2Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StackOutput4DAxis3, ClContextControlFixture, StackOutput4DAxis3Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StackOutput3DInputs3, ClContextControlFixture, StackOutput3DInputs3Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StackOutput5D, ClContextControlFixture, StackOutput5DFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StackFloat16, ClContextControlFixture, StackFloat16Test)

// Slice
ARMNN_AUTO_TEST_FIXTURE(Slice4dFloat32, ClContextControlFixture, Slice4dFloat32Test)
ARMNN_AUTO_TEST_FIXTURE(Slice3dFloat32, ClContextControlFixture, Slice3dFloat32Test)
ARMNN_AUTO_TEST_FIXTURE(Slice2dFloat32, ClContextControlFixture, Slice2dFloat32Test)
ARMNN_AUTO_TEST_FIXTURE(Slice1dFloat32, ClContextControlFixture, Slice1dFloat32Test)
ARMNN_AUTO_TEST_FIXTURE(Slice4dUint8, ClContextControlFixture, Slice4dUint8Test)
ARMNN_AUTO_TEST_FIXTURE(Slice3dUint8, ClContextControlFixture, Slice3dUint8Test)
ARMNN_AUTO_TEST_FIXTURE(Slice2dUint8, ClContextControlFixture, Slice2dUint8Test)
ARMNN_AUTO_TEST_FIXTURE(Slice1dUint8, ClContextControlFixture, Slice1dUint8Test)
ARMNN_AUTO_TEST_FIXTURE(Slice4dInt16, ClContextControlFixture, Slice4dInt16Test)
ARMNN_AUTO_TEST_FIXTURE(Slice3dInt16, ClContextControlFixture, Slice3dInt16Test)
ARMNN_AUTO_TEST_FIXTURE(Slice2dInt16, ClContextControlFixture, Slice2dInt16Test)
ARMNN_AUTO_TEST_FIXTURE(Slice1dInt16, ClContextControlFixture, Slice1dInt16Test)

// Strided Slice
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSlice4dFloat32, ClContextControlFixture, StridedSlice4dFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSlice4dReverseFloat32, ClContextControlFixture, StridedSlice4dReverseFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSliceSimpleStrideFloat32, ClContextControlFixture, StridedSliceSimpleStrideFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSliceSimpleRangeMaskFloat32, ClContextControlFixture, StridedSliceSimpleRangeMaskFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSliceShrinkAxisMaskFloat32, ClContextControlFixture, StridedSliceShrinkAxisMaskFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSliceShrinkAxisMaskCTSFloat32, ClContextControlFixture, StridedSliceShrinkAxisMaskCTSFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0Dim3Float32, ClContextControlFixture,
                     StridedSliceShrinkAxisMaskBitPosition0Dim3Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0Float32,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition1Float32,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition1Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition2Float32,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition2Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition3Float32,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition3Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And1Float32,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0And1Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And2Float32,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0And2Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And3Float32,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0And3Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And1And3Float32,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0And1And3Float32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSlice3dFloat32,
                                 ClContextControlFixture,
                                 StridedSlice3dFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSlice3dReverseFloat32, ClContextControlFixture, StridedSlice3dReverseFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSlice2dFloat32, ClContextControlFixture, StridedSlice2dFloat32Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSlice2dReverseFloat32, ClContextControlFixture, StridedSlice2dReverseFloat32Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSlice4dUint8, ClContextControlFixture, StridedSlice4dUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSlice4dReverseUint8, ClContextControlFixture, StridedSlice4dReverseUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSliceSimpleStrideUint8, ClContextControlFixture, StridedSliceSimpleStrideUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSliceSimpleRangeMaskUint8, ClContextControlFixture, StridedSliceSimpleRangeMaskUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    StridedSliceShrinkAxisMaskUint8, ClContextControlFixture, StridedSliceShrinkAxisMaskUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0Dim3Uint8,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0Dim3Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0Uint8,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition1Uint8,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition1Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition2Uint8,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition2Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition3Uint8,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition3Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And1Uint8,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0And1Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And2Uint8,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0And2Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And3Uint8,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0And3Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0And1And3Uint8,
                                 ClContextControlFixture,
                                 StridedSliceShrinkAxisMaskBitPosition0And1And3Uint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSlice3dUint8, ClContextControlFixture, StridedSlice3dUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSlice3dReverseUint8, ClContextControlFixture, StridedSlice3dReverseUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSlice2dUint8, ClContextControlFixture, StridedSlice2dUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedSlice2dReverseUint8, ClContextControlFixture, StridedSlice2dReverseUint8Test)

// Resize Bilinear - NCHW
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeBilinear,
                                 ClContextControlFixture,
                                 SimpleResizeBilinearTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeBilinearInt8,
                                 ClContextControlFixture,
                                 SimpleResizeBilinearTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeBilinearUint8,
                                 ClContextControlFixture,
                                 SimpleResizeBilinearTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearNop,
                                 ClContextControlFixture,
                                 ResizeBilinearNopTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearNopInt8,
                                 ClContextControlFixture,
                                 ResizeBilinearNopTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearNopUint8,
                                 ClContextControlFixture,
                                 ResizeBilinearNopTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearSqMin,
                                 ClContextControlFixture,
                                 ResizeBilinearSqMinTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearSqMinInt8,
                                 ClContextControlFixture,
                                 ResizeBilinearSqMinTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearSqMinUint8,
                                 ClContextControlFixture,
                                 ResizeBilinearSqMinTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearMin,
                                 ClContextControlFixture,
                                 ResizeBilinearMinTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearMinInt8,
                                 ClContextControlFixture,
                                 ResizeBilinearMinTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearMinUint8,
                                 ClContextControlFixture,
                                 ResizeBilinearMinTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeBilinear,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeBilinearTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeBilinear,
                                 ClContextControlFixture,
                                 AlignCornersResizeBilinearTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeBilinearInt8,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeBilinearTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeBilinearInt8,
                                 ClContextControlFixture,
                                 AlignCornersResizeBilinearTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeBilinearUint8,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeBilinearTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeBilinearUint8,
                                 ClContextControlFixture,
                                 AlignCornersResizeBilinearTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)

// Resize Bilinear - NHWC
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearNopNhwc,
                                 ClContextControlFixture,
                                 ResizeBilinearNopTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearNopInt8Nhwc,
                                 ClContextControlFixture,
                                 ResizeBilinearNopTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearNopUint8Nhwc,
                                 ClContextControlFixture,
                                 ResizeBilinearNopTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeBilinearNhwc,
                                 ClContextControlFixture,
                                 SimpleResizeBilinearTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeBilinearInt8Nhwc,
                                 ClContextControlFixture,
                                 SimpleResizeBilinearTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeBilinearUint8Nhwc,
                                 ClContextControlFixture,
                                 SimpleResizeBilinearTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearSqMinNhwc,
                                 ClContextControlFixture,
                                 ResizeBilinearSqMinTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearSqMinInt8Nhwc,
                                 ClContextControlFixture,
                                 ResizeBilinearSqMinTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearSqMinUint8Nhwc,
                                 ClContextControlFixture,
                                 ResizeBilinearSqMinTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearMinNhwc,
                                 ClContextControlFixture,
                                 ResizeBilinearMinTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearMinInt8Nhwc,
                                 ClContextControlFixture,
                                 ResizeBilinearMinTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeBilinearMinUint8Nhwc,
                                 ClContextControlFixture,
                                 ResizeBilinearMinTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeBilinearNhwc,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeBilinearTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeBilinearNhwc,
                                 ClContextControlFixture,
                                 AlignCornersResizeBilinearTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeBilinearInt8Nhwc,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeBilinearTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeBilinearInt8Nhwc,
                                 ClContextControlFixture,
                                 AlignCornersResizeBilinearTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeBilinearUint8Nhwc,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeBilinearTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeBilinearUint8Nhwc,
                                 ClContextControlFixture,
                                 AlignCornersResizeBilinearTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)

// Resize NearestNeighbor - NCHW
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeNearestNeighbor,
                                 ClContextControlFixture,
                                 SimpleResizeNearestNeighborTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeNearestNeighborInt8,
                                 ClContextControlFixture,
                                 SimpleResizeNearestNeighborTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeNearestNeighborUint8,
                                 ClContextControlFixture,
                                 SimpleResizeNearestNeighborTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborNop,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborNopTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborNopInt8,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborNopTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborNopUint8,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborNopTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborSqMin,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborSqMinTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborSqMinInt8,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborSqMinTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborSqMinUint8,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborSqMinTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMin,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMinTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMinInt8,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMinTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMinUint8,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMinTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMag,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMagTest<DataType::Float32>,
                                 DataLayout::NCHW, 0.1f, 50, 0.1f, 50)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMagInt8,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMagTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW, 0.1f, 50, 0.1f, 50)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMagUint8,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMagTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW, 0.1f, 50, 0.1f, 50)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeNearestNeighbour,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeNearestNeighbourTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeNearestNeighbour,
                                 ClContextControlFixture,
                                 AlignCornersResizeNearestNeighbourTest<DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeNearestNeighbourInt8,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeNearestNeighbourInt8,
                                 ClContextControlFixture,
                                 AlignCornersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeNearestNeighbourUint8,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeNearestNeighbourUint8,
                                 ClContextControlFixture,
                                 AlignCornersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                                 DataLayout::NCHW)

// Resize NearestNeighbor - NHWC
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborNopNhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborNopTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborNopInt8Nhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborNopTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborNopUint8Nhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborNopTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeNearestNeighborNhwc,
                                 ClContextControlFixture,
                                 SimpleResizeNearestNeighborTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeNearestNeighborInt8Nhwc,
                                 ClContextControlFixture,
                                 SimpleResizeNearestNeighborTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleResizeNearestNeighborUint8Nhwc,
                                 ClContextControlFixture,
                                 SimpleResizeNearestNeighborTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborSqMinNhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborSqMinTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborSqMinInt8Nhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborSqMinTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborSqMinUint8Nhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborSqMinTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMinNhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMinTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMinInt8Nhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMinTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMinUint8Nhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMinTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMagNhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMagTest<DataType::Float32>,
                                 DataLayout::NHWC, 0.1f, 50, 0.1f, 50)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMagInt8Nhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMagTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC, 0.1f, 50, 0.1f, 50)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ResizeNearestNeighborMagUint8Nhwc,
                                 ClContextControlFixture,
                                 ResizeNearestNeighborMagTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC, 0.1f, 50, 0.1f, 50)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeNearestNeighbourNhwc,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeNearestNeighbourTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeNearestNeighbourNhwc,
                                 ClContextControlFixture,
                                 AlignCornersResizeNearestNeighbourTest<DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeNearestNeighbourInt8Nhwc,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeNearestNeighbourInt8Nhwc,
                                 ClContextControlFixture,
                                 AlignCornersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(HalfPixelCentersResizeNearestNeighbourUint8Nhwc,
                                 ClContextControlFixture,
                                 HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AlignCornersResizeNearestNeighbourUint8Nhwc,
                                 ClContextControlFixture,
                                 AlignCornersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                                 DataLayout::NHWC)

// Rsqrt
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Rsqrt2d, ClContextControlFixture, Rsqrt2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Rsqrt3d, ClContextControlFixture, Rsqrt3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RsqrtZero, ClContextControlFixture, RsqrtZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(RsqrtNegative, ClContextControlFixture, RsqrtNegativeTest<DataType::Float32>)

// Sqrt
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Sqrt2d, ClContextControlFixture, Sqrt2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Sqrt3d, ClContextControlFixture, Sqrt3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SqrtZero, ClContextControlFixture, SqrtZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SqrtNegative, ClContextControlFixture, SqrtNegativeTest<DataType::Float32>)

// Quantize
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(QuantizeSimpleUint8, ClContextControlFixture, QuantizeSimpleUint8Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(QuantizeClampUint8, ClContextControlFixture, QuantizeClampUint8Test)

// Dequantize
ARMNN_AUTO_TEST_FIXTURE(DequantizeSimpleUint8, ClContextControlFixture, DequantizeSimpleUint8Test)
ARMNN_AUTO_TEST_FIXTURE(DequantizeOffsetUint8, ClContextControlFixture, DequantizeOffsetUint8Test)
ARMNN_AUTO_TEST_FIXTURE(DequantizeSimpleInt16, ClContextControlFixture, DequantizeSimpleInt16Test)
ARMNN_AUTO_TEST_FIXTURE(DequantizeSimpleUint8ToFp16, ClContextControlFixture, DequantizeSimpleUint8ToFp16Test)
ARMNN_AUTO_TEST_FIXTURE(DequantizeSimpleInt16ToFp16, ClContextControlFixture, DequantizeSimpleInt16ToFp16Test)

// Transpose
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SimpleTransposeFloat32, ClContextControlFixture, SimpleTransposeTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeFloat32ValueSet1Test, ClContextControlFixture, TransposeValueSet1Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeFloat32ValueSet2Test, ClContextControlFixture, TransposeValueSet2Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeFloat32ValueSet3Test, ClContextControlFixture, TransposeValueSet3Test<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SimpleTransposeQASymmS8, ClContextControlFixture, SimpleTransposeTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeQASymmS8ValueSet1Test, ClContextControlFixture, TransposeValueSet1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeQASymmS8ValueSet2Test, ClContextControlFixture, TransposeValueSet2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeQASymmS8ValueSet3Test, ClContextControlFixture, TransposeValueSet3Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SimpleTransposeQASymm8, ClContextControlFixture, SimpleTransposeTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeQASymm8ValueSet1Test, ClContextControlFixture, TransposeValueSet1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeQASymm8ValueSet2Test, ClContextControlFixture, TransposeValueSet2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeQASymm8ValueSet3Test, ClContextControlFixture, TransposeValueSet3Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    SimpleTransposeQSymm16, ClContextControlFixture, SimpleTransposeTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeQSymm16ValueSet1Test, ClContextControlFixture, TransposeValueSet1Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeQSymm16ValueSet2Test, ClContextControlFixture, TransposeValueSet2Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    TransposeQSymm16ValueSet3Test, ClContextControlFixture, TransposeValueSet3Test<DataType::QSymmS16>)

// TransposeConvolution2d
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleTransposeConvolution2dFloatNchw,
                                 ClContextControlFixture,
                                 SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleTransposeConvolution2dFloatNhwc,
                                 ClContextControlFixture,
                                 SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleTransposeConvolution2dUint8Nchw,
                                 ClContextControlFixture,
                                 SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SimpleTransposeConvolution2dUint8Nhwc,
                                 ClContextControlFixture,
                                 SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedSimpleTransposeConvolution2dFloatNchw,
                                 ClContextControlFixture,
                                 SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedSimpleTransposeConvolution2dFloatNhwc,
                                 ClContextControlFixture,
                                 SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedSimpleTransposeConvolution2dUint8Nchw,
                                 ClContextControlFixture,
                                 SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedSimpleTransposeConvolution2dUint8Nhwc,
                                 ClContextControlFixture,
                                 SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PaddedTransposeConvolution2dFloatNchw,
                                 ClContextControlFixture,
                                 PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PaddedTransposeConvolution2dFloatNhwc,
                                 ClContextControlFixture,
                                 PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PaddedTransposeConvolution2dUint8Nchw,
                                 ClContextControlFixture,
                                 PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(PaddedTransposeConvolution2dUint8Nhwc,
                                 ClContextControlFixture,
                                 PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedPaddedTransposeConvolution2dFloatNchw,
                                 ClContextControlFixture,
                                 PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedPaddedTransposeConvolution2dFloatNhwc,
                                 ClContextControlFixture,
                                 PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedPaddedTransposeConvolution2dUint8Nchw,
                                 ClContextControlFixture,
                                 PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedPaddedTransposeConvolution2dUint8Nhwc,
                                 ClContextControlFixture,
                                 PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedTransposeConvolution2dFloatNchw,
                                 ClContextControlFixture,
                                 StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedTransposeConvolution2dFloatNhwc,
                                 ClContextControlFixture,
                                 StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedTransposeConvolution2dUint8Nchw,
                                 ClContextControlFixture,
                                 StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(StridedTransposeConvolution2dUint8Nhwc,
                                 ClContextControlFixture,
                                 StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedStridedTransposeConvolution2dFloatNchw,
                                 ClContextControlFixture,
                                 StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 false,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedStridedTransposeConvolution2dFloatNhwc,
                                 ClContextControlFixture,
                                 StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 true,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedStridedTransposeConvolution2dUint8Nchw,
                                 ClContextControlFixture,
                                 StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(UnbiasedStridedTransposeConvolution2dUint8Nhwc,
                                 ClContextControlFixture,
                                 StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 true,
                                 DataLayout::NHWC)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MultiChannelTransposeConvolution2dFloatNchw,
                                 ClContextControlFixture,
                                 MultiChannelTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MultiChannelTransposeConvolution2dFloatNhwc,
                                 ClContextControlFixture,
                                 MultiChannelTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                                 DataLayout::NHWC)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MultiChannelTransposeConvolution2dUint8Nchw,
                                 ClContextControlFixture,
                                 MultiChannelTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 DataLayout::NCHW)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(MultiChannelTransposeConvolution2dUint8Nhwc,
                                 ClContextControlFixture,
                                 MultiChannelTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                                 DataLayout::NHWC)

// Abs
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Abs2d, ClContextControlFixture, Abs2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Abs3d, ClContextControlFixture, Abs3dTest<DataType::Float32>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AbsZero, ClContextControlFixture, AbsZeroTest<DataType::Float32>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Abs2dFloat16, ClContextControlFixture, Abs2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Abs3dFloat16, ClContextControlFixture, Abs3dTest<DataType::Float16>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(AbsZeroFloat16, ClContextControlFixture, AbsZeroTest<DataType::Float16>)

// ArgMinMax
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMinFloat32, ClContextControlFixture, ArgMinSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMaxFloat32, ClContextControlFixture, ArgMaxSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMinChannel, ClContextControlFixture, ArgMinChannelTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMaxChannel, ClContextControlFixture, ArgMaxChannelTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMaxHeight, ClContextControlFixture, ArgMaxHeightTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMinWidth, ClContextControlFixture, ArgMinWidthTest<DataType::Float32>)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMinQAsymm8, ClContextControlFixture, ArgMinSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMaxQAsymm8, ClContextControlFixture, ArgMaxSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMinChannelQAsymm8, ClContextControlFixture, ArgMinChannelTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMaxChannelQAsymm8, ClContextControlFixture, ArgMaxChannelTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMaxHeightQAsymm8, ClContextControlFixture, ArgMaxHeightTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ArgMinWidthQAsymm8, ClContextControlFixture, ArgMinWidthTest<DataType::QAsymmU8>)

// Neg
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Neg2d, ClContextControlFixture, Neg2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Neg3d, ClContextControlFixture, Neg3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(NegZero, ClContextControlFixture, NegZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(NegNegative, ClContextControlFixture, NegNegativeTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Neg2dFloat16, ClContextControlFixture, Neg2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Neg3dFloat16, ClContextControlFixture, Neg3dTest<DataType::Float16>)

// Exp
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Exp2d, ClContextControlFixture, Exp2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Exp3d, ClContextControlFixture, Exp3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ExpZero, ClContextControlFixture, ExpZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ExpNegative, ClContextControlFixture, ExpNegativeTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Exp2dFloat16, ClContextControlFixture, Exp2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Exp3dFloat16, ClContextControlFixture, Exp3dTest<DataType::Float16>)

// Sin
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Sin2d, ClContextControlFixture, Sin2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Sin3d, ClContextControlFixture, Sin3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SinZero, ClContextControlFixture, SinZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(SinNegative, ClContextControlFixture, SinNegativeTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Sin2dFloat16, ClContextControlFixture, Sin2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Sin3dFloat16, ClContextControlFixture, Sin3dTest<DataType::Float16>)

// Log
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Log2d, ClContextControlFixture, Log2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Log3d, ClContextControlFixture, Log3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogZero, ClContextControlFixture, LogZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogNegative, ClContextControlFixture, LogNegativeTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Log2dFloat16, ClContextControlFixture, Log2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(Log3dFloat16, ClContextControlFixture, Log3dTest<DataType::Float16>)

// Logical
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalNot, ClContextControlFixture, LogicalNotTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalNotInt, ClContextControlFixture, LogicalNotIntTest)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalAnd, ClContextControlFixture, LogicalAndTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalAndInt, ClContextControlFixture, LogicalAndIntTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalAndBroadcast1, ClContextControlFixture, LogicalAndBroadcast1Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalAndBroadcast2, ClContextControlFixture, LogicalAndBroadcast2Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalAndBroadcast3, ClContextControlFixture, LogicalAndBroadcast3Test)

ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalOr, ClContextControlFixture, LogicalOrTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalOrInt, ClContextControlFixture, LogicalOrIntTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalOrBroadcast1, ClContextControlFixture, LogicalOrBroadcast1Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalOrBroadcast2, ClContextControlFixture, LogicalOrBroadcast2Test)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(LogicalOrBroadcast3, ClContextControlFixture, LogicalOrBroadcast3Test)

// ReduceSum
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReduceSumFloat32, ClContextControlFixture, ReduceSumSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    ReduceSumSingleAxisFloat32_1, ClContextControlFixture, ReduceSumSingleAxisTest1<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    ReduceSumSingleAxisFloat32_2, ClContextControlFixture, ReduceSumSingleAxisTest2<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    ReduceSumSingleAxisFloat32_3, ClContextControlFixture, ReduceSumSingleAxisTest3<DataType::Float32>)

// ReduceProd
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReduceProdFloat32, ClContextControlFixture, ReduceProdSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    ReduceProdSingleAxisFloat32_1, ClContextControlFixture, ReduceProdSingleAxisTest1<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    ReduceProdSingleAxisFloat32_2, ClContextControlFixture, ReduceProdSingleAxisTest2<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    ReduceProdSingleAxisFloat32_3, ClContextControlFixture, ReduceProdSingleAxisTest3<DataType::Float32>)

// ReduceMax
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReduceMaxFloat32, ClContextControlFixture, ReduceMaxSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    ReduceMaxNegativeAxisFloat32, ClContextControlFixture, ReduceMaxNegativeAxisTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReduceMax2Float32, ClContextControlFixture, ReduceMaxSimpleTest2<DataType::Float32>)

// ReduceMin
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(ReduceMinFloat32, ClContextControlFixture, ReduceMinSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(
    ReduceMinNegativeAxisFloat32, ClContextControlFixture, ReduceMinNegativeAxisTest<DataType::Float32>)

// Cast
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(CastInt32ToFloat, ClContextControlFixture, CastInt32ToFloat2dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(CastFloat16ToFloat32, ClContextControlFixture, CastFloat16ToFloat322dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(CastFloatToFloat16, ClContextControlFixture, CastFloat32ToFloat162dTest)
ARMNN_AUTO_TEST_FIXTURE_WITH_THF(CastFloatToUInt8, ClContextControlFixture, CastFloat32ToUInt82dTest)

// ChannelShuffle
ARMNN_AUTO_TEST_CASE_WITH_THF(ChannelShuffle4DFloat32, ChannelShuffle4DTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ChannelShuffle4DQAsymmU8, ChannelShuffle4DTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ChannelShuffle4DQAsymmS8, ChannelShuffle4DTest<DataType::QAsymmS8>)

#if defined(ARMNNREF_ENABLED)

TEST_CASE_FIXTURE(ClContextControlFixture, "ClContextControlFixture") {}

// The ARMNN_COMPARE_REF_AUTO_TEST_CASE and the ARMNN_COMPARE_REF_FIXTURE_TEST_CASE test units are not available
// if the reference backend is not built

// COMPARE tests

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxBeta1WithReference, CompareSoftmaxTest, 1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxBeta2WithReference, CompareSoftmaxTest, 2.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxUint8, CompareSoftmaxUint8Test, 1.0f)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareConv2dWithReference, CompareConvolution2dTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareDepthwiseConv2dWithReferenceFloat32,
                                 CompareDepthwiseConvolution2dFloatTest,
                                 DataLayout::NCHW)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareDepthwiseConv2dWithReferenceUint8,
                                 CompareDepthwiseConvolution2dUint8Test,
                                 DataLayout::NCHW)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareDepthwiseConv2dWithReferenceFloat32Nhwc,
                                 CompareDepthwiseConvolution2dFloatTest,
                                 DataLayout::NHWC)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareDepthwiseConv2dWithReferenceUint8Nhwc,
                                 CompareDepthwiseConvolution2dUint8Test,
                                 DataLayout::NHWC)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareNormalizationWithinWithReference, CompareNormalizationTest,
                                          NormalizationAlgorithmChannel::Within,
                                          NormalizationAlgorithmMethod::LocalBrightness)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareNormalizationAcrossWithReference, CompareNormalizationTest,
                                          NormalizationAlgorithmChannel::Across,
                                          NormalizationAlgorithmMethod::LocalBrightness)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareMaxPooling2dWithRef, ComparePooling2dTest, PoolingAlgorithm::Max)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareAveragePooling2dWithRef,
                                          ComparePooling2dTest, PoolingAlgorithm::Average)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareAveragePooling2dWithRefUint8, ComparePooling2dUint8Test,
                                          PoolingAlgorithm::Average)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareL2Pooling2dWithRef, ComparePooling2dTest, PoolingAlgorithm::L2)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareMaxPooling3dWithRef, ComparePooling3dTest, PoolingAlgorithm::Max,
                                          DataLayout::NDHWC)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareAveragePooling3dWithRef, ComparePooling3dTest,
                                          PoolingAlgorithm::Average, DataLayout::NDHWC)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareL2Pooling3dWithRef, ComparePooling3dTest, PoolingAlgorithm::L2,
                                          DataLayout::NDHWC)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareAddition, CompareAdditionTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareMultiplicationWithRef, CompareMultiplicationTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareBatchNorm, CompareBatchNormTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareReLu1, CompareBoundedReLuTest, 1.0f, -1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareReLu6, CompareBoundedReLuTest, 6.0f, 0.0f)

// ============================================================================
// FIXTURE tests

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareSigmoidActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::Sigmoid, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareTanhActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::TanH, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareLinearActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::Linear, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareReLuActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::ReLu, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareBoundedReLuActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::BoundedReLu, 5u)
ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareBoundedReLuActivationWithReferenceUint8, ActivationFixture,
                                    CompareActivationUint8Test, ActivationFunction::BoundedReLu)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareSoftReLuActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::SoftReLu, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareLeakyReLuActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::LeakyReLu, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareAbsActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::Abs, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareSqrtActivationWithReference, PositiveActivationFixture,
                                    CompareActivationTest, ActivationFunction::Sqrt, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareSquareActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::Square, 5u)

ARMNN_COMPARE_REF_FIXTURE_TEST_CASE_WITH_THF(CompareEluActivationWithReference, ActivationFixture,
                                    CompareActivationTest, ActivationFunction::Elu, 5u)

#endif

}
