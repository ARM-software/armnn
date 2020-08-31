//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "RefWorkloadFactoryHelper.hpp"

#include <backendsCommon/test/LayerTests.hpp>

#include <reference/RefWorkloadFactory.hpp>

#include <test/UnitTests.hpp>

#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Compute_Reference)

using namespace armnn;

using FactoryType = RefWorkloadFactory;

// ============================================================================
// UNIT tests

// Convolution
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5, SimpleConvolution2d3x5Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5Uint8, SimpleConvolution2d3x5Uint8Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5Nhwc, SimpleConvolution2d3x5Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5Uint8Nhwc, SimpleConvolution2d3x5Uint8Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5QSymm16, SimpleConvolution2d3x5QSymm16Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x5QSymm16Nhwc,
                              SimpleConvolution2d3x5QSymm16Test,
                              true,
                              DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2d, SimpleConvolution2d3x5Test, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolutionUint8, SimpleConvolution2d3x5Uint8Test, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dNhwc, SimpleConvolution2d3x5Test, false, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolutionUint8Nhwc, SimpleConvolution2d3x5Uint8Test, false, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution1d, Convolution1dTest, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution1dUint8, Convolution1dUint8Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3, SimpleConvolution2d3x3Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3Uint8, SimpleConvolution2d3x3Uint8Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3QSymm16, SimpleConvolution2d3x3QSymm16Test, true, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3Nhwc, SimpleConvolution2d3x3Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3Uint8Nhwc, SimpleConvolution2d3x3Uint8Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3QSymm16Nhwc, SimpleConvolution2d3x3QSymm16Test, true,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dSquare, SimpleConvolution2d3x3Test, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dSquareNhwc, SimpleConvolution2d3x3Test, false, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dSquareStride2x2Nhwc,
                     SimpleConvolution2d3x3Stride2x2Test,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPaddingLargerThanHalfKernelSize,
                     Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPadding,
                              Convolution2dAsymmetricPaddingTest, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPaddingLargerThanHalfKernelSizeNhwc,
                     Convolution2dAsymmetricPaddingLargerThanHalfKernelSizeTest,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPaddingNhwc,
                     Convolution2dAsymmetricPaddingTest,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dSquareNhwc, SimpleConvolution2d3x3NhwcTest, false)

ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3BFloat16,
                     Convolution2d3x3Dilation3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3NhwcBFloat16,
                     Convolution2d3x3Dilation3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3,
                     Convolution2d3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3Nhwc,
                     Convolution2d3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3Int8,
                     Convolution2d3x3Dilation3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3NhwcInt8,
                     Convolution2d3x3Dilation3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3Uint8,
                     Convolution2d3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3NhwcUint8,
                     Convolution2d3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3Int16,
                     Convolution2d3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Dilation3x3NhwcInt16,
                     Convolution2d3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3BFloat16,
                     Convolution2d2x3x3Dilation3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3NhwcBFloat16,
                     Convolution2d2x3x3Dilation3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3,
                     Convolution2d2x3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3Nhwc,
                     Convolution2d2x3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3Int8,
                     Convolution2d2x3x3Dilation3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3NhwcInt8,
                     Convolution2d2x3x3Dilation3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3Uint8,
                     Convolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3NhwcUint8,
                     Convolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3Int16,
                     Convolution2d2x3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x3x3Dilation3x3NhwcInt16,
                     Convolution2d2x3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3BFloat16,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3NhwcBFloat16,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Nhwc,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Int8,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3NhwcInt8,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Uint8,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3NhwcUint8,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Int16,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3NhwcInt16,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2dPerAxisQuantTestNchw, Convolution2dPerAxisQuantTest, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2dPerAxisQuantTestNhwc, Convolution2dPerAxisQuantTest, DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Stride2x2Bf16,
                              Convolution2d3x3Stride2x2BFloat16Test,
                              false,
                              DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d3x3Stride2x2BFloat16SmallValue,
                     Convolution2d3x3Stride2x2BFloat16SmallValueTest,
                     false,
                     DataLayout::NHWC);

// Depthwise Convolution
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d, DepthwiseConvolution2dTest, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dUint8, DepthwiseConvolution2dUint8Test, true, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2d, DepthwiseConvolution2dTest, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dUint8,
                     DepthwiseConvolution2dUint8Test,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dQSymm16, DepthwiseConvolution2dInt16Test, true, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dNhwc, DepthwiseConvolution2dTest, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dUint8Nhwc, DepthwiseConvolution2dUint8Test, true, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dNhwc, DepthwiseConvolution2dTest, false, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dUint8Nhwc,
                     DepthwiseConvolution2dUint8Test,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthNhwc, DepthwiseConvolution2dDepthNhwcTest, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleDepthwiseConvolution2d3x3Dilation3x3Nhwc,
                     SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3Nhwc,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3BFloat16,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3NhwcBFloat16,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3Int8,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3NhwcInt8,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3Uint8,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3NhwcUint8,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3Int16,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d3x3Dilation3x3NhwcInt16,
                     DepthwiseConvolution2d3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3Nhwc,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3BFloat16,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3NhwcBFloat16,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::BFloat16, DataType::BFloat16>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3Int8,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3NhwcInt8,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3Uint8,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3NhwcUint8,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3Int16,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d2x3x3Dilation3x3NhwcInt16,
                     DepthwiseConvolution2d2x3x3Dilation3x3Test<DataType::QSymmS16, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dMult4,
                     DepthwiseConvolution2dMult4Test<armnn::DataType::Float32, armnn::DataType::Float32>,
                     false,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dMult2,
                     DepthwiseConvolution2dMult2Test<armnn::DataType::Float32, armnn::DataType::Float32>,
                     false,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dMult4BFloat16,
                     DepthwiseConvolution2dMult4Test<armnn::DataType::BFloat16, armnn::DataType::BFloat16>,
                     false,
                     armnn::DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dMult2BFloat16,
                     DepthwiseConvolution2dMult2Test<armnn::DataType::BFloat16, armnn::DataType::BFloat16>,
                     false,
                     armnn::DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Int16,
                     DepthwiseConvolution2dDepthMul1Int16Test, true, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test, false, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Nhwc,
                     DepthwiseConvolution2dDepthMul1Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test, true, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Nhwc,
                     DepthwiseConvolution2dDepthMul1Test, false, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test, false, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dAsymmetric,
                     DepthwiseConvolution2dAsymmetricTest, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dAsymmetric,
                     DepthwiseConvolution2dAsymmetricTest, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dAsymmetricNhwc,
                     DepthwiseConvolution2dAsymmetricTest, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dAsymmetricNhwc,
                     DepthwiseConvolution2dAsymmetricTest, false, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul64, DepthwiseConvolution2dDepthMul64Test);

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dPerAxisQuantTestNchw, DepthwiseConvolution2dPerAxisQuantTest,
                     DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dPerAxisQuantTestNhwc, DepthwiseConvolution2dPerAxisQuantTest,
                     DataLayout::NHWC);

// Pooling
//MaxPooling
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize2x2Stride2x2, SimpleMaxPooling2dSize2x2Stride2x2Test, false)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize2x2Stride2x2Uint8, SimpleMaxPooling2dSize2x2Stride2x2Uint8Test, false)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize2x2Stride2x2Int16, SimpleMaxPooling2dSize2x2Stride2x2Int16Test, false)

ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4, SimpleMaxPooling2dSize3x3Stride2x4Test, false)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4Uint8, SimpleMaxPooling2dSize3x3Stride2x4Uint8Test, false)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dSize3x3Stride2x4Int16, SimpleMaxPooling2dSize3x3Stride2x4Int16Test, false)

ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2d, SimpleMaxPooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dNhwc, SimpleMaxPooling2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dUint8, SimpleMaxPooling2dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dInt16, SimpleMaxPooling2dInt16Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dUint8Nhwc, SimpleMaxPooling2dUint8Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleMaxPooling2dInt16Nhwc, SimpleMaxPooling2dInt16Test, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleMaxPooling2d, IgnorePaddingSimpleMaxPooling2dTest)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleMaxPooling2dUint8, IgnorePaddingSimpleMaxPooling2dUint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingSimpleMaxPooling2dInt16, IgnorePaddingSimpleMaxPooling2dInt16Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingMaxPooling2dSize3, IgnorePaddingMaxPooling2dSize3Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingMaxPooling2dSize3Uint8, IgnorePaddingMaxPooling2dSize3Uint8Test)
ARMNN_AUTO_TEST_CASE(IgnorePaddingMaxPooling2dSize3Int16, IgnorePaddingMaxPooling2dSize3Int16Test)

//AveragePooling
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2d, SimpleAveragePooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dNhwc, SimpleAveragePooling2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dUint8, SimpleAveragePooling2dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dInt16, SimpleAveragePooling2dInt16Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dUint8Nhwc, SimpleAveragePooling2dUint8Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleAveragePooling2dInt16Nhwc, SimpleAveragePooling2dInt16Test, DataLayout::NHWC)

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

ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2d, SimpleL2Pooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dNhwc, SimpleL2Pooling2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dUint8, SimpleL2Pooling2dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dInt16, SimpleL2Pooling2dInt16Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dNhwcUint8, SimpleL2Pooling2dUint8Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleL2Pooling2dNhwcInt16, SimpleL2Pooling2dInt16Test, DataLayout::NHWC)

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

// InstanceNormalization
ARMNN_AUTO_TEST_CASE(InstanceNormFloat32Nchw, InstanceNormFloat32Test, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(InstanceNormFloat16Nchw, InstanceNormFloat16Test, DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(InstanceNormFloat32Nhwc, InstanceNormFloat32Test, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(InstanceNormFloat16Nhwc, InstanceNormFloat16Test, DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE(InstanceNormFloat32Nchw2, InstanceNormFloat32Test2, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(InstanceNormFloat16Nchw2, InstanceNormFloat16Test2, DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(InstanceNormFloat32Nhwc2, InstanceNormFloat32Test2, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(InstanceNormFloat16Nhwc2, InstanceNormFloat16Test2, DataLayout::NHWC);

// Normalization
ARMNN_AUTO_TEST_CASE(SimpleNormalizationAcross, SimpleNormalizationAcrossTest)
ARMNN_AUTO_TEST_CASE(SimpleNormalizationWithin, SimpleNormalizationWithinTest)
ARMNN_AUTO_TEST_CASE(SimpleNormalizationAcrossNhwc, SimpleNormalizationAcrossNhwcTest)

// Softmax
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta1, SimpleSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta2, SimpleSoftmaxTest, 2.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta1Uint8, SimpleSoftmaxUint8Test, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta2Uint8, SimpleSoftmaxUint8Test, 2.0f)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dSoftmax, Simple3dSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dSoftmaxUint8, Simple3dSoftmaxUint8Test, 1.0f)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dSoftmax, Simple4dSoftmaxTest, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dSoftmaxUint8, Simple4dSoftmaxUint8Test, 1.0f)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxFloat16, SimpleSoftmaxFloat16Test, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dSoftmaxFloat16, Simple3dSoftmaxFloat16Test, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dSoftmaxFloat16, Simple4dSoftmaxFloat16Test, 1.0f)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxUint16, SimpleSoftmaxUint16Test, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dSoftmaxUint16, Simple3dSoftmaxUint16Test, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dSoftmaxUint16, Simple4dSoftmaxUint16Test, 1.0f)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple2dAxis0Softmax, SimpleAxisSoftmaxTest, 1.0f, 0)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple2dAxis1Softmax, SimpleAxisSoftmaxTest, 1.0f, 1)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple2dAxis0NegSoftmax, SimpleAxisSoftmaxTest, 1.0f, -2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple2dAxis1NegSoftmax, SimpleAxisSoftmaxTest, 1.0f, -1)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis0Softmax, Simple3dAxisSoftmaxTest, 1.0f, 0)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis1Softmax, Simple3dAxisSoftmaxTest, 1.0f, 1)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis2Softmax, Simple3dAxisSoftmaxTest, 1.0f, 2)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis0NegSoftmax, Simple3dAxisSoftmaxTest, 1.0f, -3)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis1NegSoftmax, Simple3dAxisSoftmaxTest, 1.0f, -2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple3dAxis2NegSoftmax, Simple3dAxisSoftmaxTest, 1.0f, -1)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis0Softmax, Simple4dAxisSoftmaxTest, 1.0f, 0)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis1Softmax, Simple4dAxisSoftmaxTest, 1.0f, 1)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis2Softmax, Simple4dAxisSoftmaxTest, 1.0f, 2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis3Softmax, Simple4dAxisSoftmaxTest, 1.0f, 3)

ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis0NegSoftmax, Simple4dAxisSoftmaxTest, 1.0f, -4)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis1NegSoftmax, Simple4dAxisSoftmaxTest, 1.0f, -3)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis2NegSoftmax, Simple4dAxisSoftmaxTest, 1.0f, -2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Simple4dAxis3NegSoftmax, Simple4dAxisSoftmaxTest, 1.0f, -1)

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
ARMNN_AUTO_TEST_CASE(SqrtNN, SqrtNNTest)
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

// Elu Activation
ARMNN_AUTO_TEST_CASE(Elu, EluTest)
ARMNN_AUTO_TEST_CASE(EluUint8, EluUint8Test)
ARMNN_AUTO_TEST_CASE(EluInt16, EluInt16Test)
// HardSwish Activation
ARMNN_AUTO_TEST_CASE(HardSwish, HardSwishTest)
ARMNN_AUTO_TEST_CASE(HardSwishUint8, HardSwishUint8Test)
ARMNN_AUTO_TEST_CASE(HardSwishInt16, HardSwishInt16Test)

// Fully Connected
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFullyConnected, FullyConnectedFloat32Test, false, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedUint8, FullyConnectedTest<DataType::QAsymmU8>, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedQSymm16, FullyConnectedTest<DataType::QSymmS16>, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFullyConnectedWithBias, FullyConnectedFloat32Test, true, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedBiasedUint8, FullyConnectedTest<DataType::QAsymmU8>, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedBiasedQSymm16, FullyConnectedTest<DataType::QSymmS16>, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFullyConnectedWithTranspose, FullyConnectedFloat32Test, false, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedLarge, FullyConnectedLargeTest, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedLargeTransposed, FullyConnectedLargeTest, true)

// Splitter
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterFloat32, SplitterFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterFloat16, SplitterFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterUint8, SplitterUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterInt16, SplitterInt16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterFloat32, CopyViaSplitterFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterFloat16, CopyViaSplitterFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterUint8, CopyViaSplitterUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterInt16, CopyViaSplitterInt16Test)

// Concat
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConcat, ConcatTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatBFloat16, ConcatBFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatFloat16, ConcatFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint8, ConcatUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint8DifferentQParams, ConcatUint8DifferentQParamsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint16, ConcatUint16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint8DifferentInputOutputQParam,
                     ConcatDifferentInputOutputQParamTest<DataType::QAsymmU8>, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatInt16DifferentInputOutputQParam,
                     ConcatDifferentInputOutputQParamTest<DataType::QSymmS16>, true)

// Add
ARMNN_AUTO_TEST_CASE(SimpleAdd, AdditionTest)
ARMNN_AUTO_TEST_CASE(Add5d, Addition5dTest)
ARMNN_AUTO_TEST_CASE(AddBroadcast1Element, AdditionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(AddBroadcast, AdditionBroadcastTest)

ARMNN_AUTO_TEST_CASE(AdditionUint8, AdditionUint8Test)
ARMNN_AUTO_TEST_CASE(AddBroadcastUint8, AdditionBroadcastUint8Test)
ARMNN_AUTO_TEST_CASE(AddBroadcast1ElementUint8, AdditionBroadcast1ElementUint8Test)

ARMNN_AUTO_TEST_CASE(AdditionInt16, AdditionInt16Test)
ARMNN_AUTO_TEST_CASE(AddBroadcastInt16, AdditionBroadcastInt16Test)
ARMNN_AUTO_TEST_CASE(AddBroadcast1ElementInt16, AdditionBroadcast1ElementInt16Test)

ARMNN_AUTO_TEST_CASE(AdditionInt32, AdditionInt32Test)
ARMNN_AUTO_TEST_CASE(AddBroadcastInt32, AdditionBroadcastInt32Test)
ARMNN_AUTO_TEST_CASE(AddBroadcast1ElementInt32, AdditionBroadcast1ElementInt32Test)

// Sub
ARMNN_AUTO_TEST_CASE(SimpleSub, SubtractionTest)
ARMNN_AUTO_TEST_CASE(SubBroadcast1Element, SubtractionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(SubBroadcast, SubtractionBroadcastTest)

ARMNN_AUTO_TEST_CASE(SimpleSubFloat16, SubtractionTest)
ARMNN_AUTO_TEST_CASE(SubBroadcast1ElementFloat16, SubtractionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(SubBroadcastFloat16, SubtractionBroadcastTest)

ARMNN_AUTO_TEST_CASE(SubtractionUint8, SubtractionUint8Test)
ARMNN_AUTO_TEST_CASE(SubBroadcastUint8, SubtractionBroadcastUint8Test)
ARMNN_AUTO_TEST_CASE(SubBroadcast1ElementUint8, SubtractionBroadcast1ElementUint8Test)

ARMNN_AUTO_TEST_CASE(SubtractionInt16, SubtractionInt16Test)
ARMNN_AUTO_TEST_CASE(SubBroadcastInt16, SubtractionBroadcastInt16Test)
ARMNN_AUTO_TEST_CASE(SubBroadcast1ElementInt16, SubtractionBroadcast1ElementInt16Test)

ARMNN_AUTO_TEST_CASE(SubtractionInt32, SubtractionInt32Test)
ARMNN_AUTO_TEST_CASE(SubBroadcastInt32, SubtractionBroadcastInt32Test)
ARMNN_AUTO_TEST_CASE(SubBroadcast1ElementInt32, SubtractionBroadcast1ElementInt32Test)

// Div
ARMNN_AUTO_TEST_CASE(SimpleDivision, DivisionTest)
ARMNN_AUTO_TEST_CASE(DivisionByZero, DivisionByZeroTest)
ARMNN_AUTO_TEST_CASE(DivisionBroadcast1Element, DivisionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(DivisionBroadcast1DVector, DivisionBroadcast1DVectorTest)

ARMNN_AUTO_TEST_CASE(DivisionFloat16, DivisionFloat16Test)
ARMNN_AUTO_TEST_CASE(DivisionFloat16Broadcast1Element, DivisionBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE(DivisionFloat16Broadcast1DVector, DivisionBroadcast1DVectorFloat16Test)

// NOTE: division by zero for quantized div needs more attention
//       see IVGCVSW-1849
ARMNN_AUTO_TEST_CASE(DivisionUint8, DivisionUint8Test)
ARMNN_AUTO_TEST_CASE(DivisionUint8Broadcast1Element, DivisionBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(DivisionUint8Broadcast1DVector, DivisionBroadcast1DVectorUint8Test)

ARMNN_AUTO_TEST_CASE(DivisionInt16, DivisionInt16Test)
ARMNN_AUTO_TEST_CASE(DivisionInt16Broadcast1Element, DivisionBroadcast1ElementInt16Test)
ARMNN_AUTO_TEST_CASE(DivisionInt16Broadcast1DVector, DivisionBroadcast1DVectorInt16Test)

ARMNN_AUTO_TEST_CASE(DivisionInt32, DivisionInt32Test)
ARMNN_AUTO_TEST_CASE(DivisionInt32Broadcast1Element, DivisionBroadcast1ElementInt32Test)
ARMNN_AUTO_TEST_CASE(DivisionInt32Broadcast1DVector, DivisionBroadcast1DVectorInt32Test)

// Equal
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualSimple,            EqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1Element, EqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1dVector, EqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(EqualSimpleFloat16,            EqualSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1ElementFloat16, EqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1dVectorFloat16, EqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(EqualSimpleUint8,            EqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1ElementUint8, EqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1dVectorUint8, EqualBroadcast1dVectorUint8Test)

// Greater
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterSimple,            GreaterSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1Element, GreaterBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1dVector, GreaterBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterSimpleFloat16,            GreaterSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1ElementFloat16, GreaterBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1dVectorFloat16, GreaterBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterSimpleUint8,            GreaterSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1ElementUint8, GreaterBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1dVectorUint8, GreaterBroadcast1dVectorUint8Test)

// GreaterOrEqual
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualSimple,            GreaterOrEqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1Element, GreaterOrEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1dVector, GreaterOrEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualSimpleFloat16,            GreaterOrEqualSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1ElementFloat16, GreaterOrEqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1dVectorFloat16, GreaterOrEqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualSimpleUint8,            GreaterOrEqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1ElementUint8, GreaterOrEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1dVectorUint8, GreaterOrEqualBroadcast1dVectorUint8Test)

// Less
ARMNN_AUTO_TEST_CASE_WITH_THF(LessSimple,            LessSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1Element, LessBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1dVector, LessBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessSimpleFloat16,            LessSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1ElementFloat16, LessBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1dVectorFloat16, LessBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessSimpleUint8,            LessSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1ElementUint8, LessBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1dVectorUint8, LessBroadcast1dVectorUint8Test)

// LessOrEqual
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualSimple,            LessOrEqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1Element, LessOrEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1dVector, LessOrEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualSimpleFloat16,            LessOrEqualSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1ElementFloat16, LessOrEqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1dVectorFloat16, LessOrEqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualSimpleUint8,            LessOrEqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1ElementUint8, LessOrEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1dVectorUint8, LessOrEqualBroadcast1dVectorUint8Test)

// NotEqual
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualSimple,            NotEqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1Element, NotEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1dVector, NotEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualSimpleFloat16,            NotEqualSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1ElementFloat16, NotEqualBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1dVectorFloat16, NotEqualBroadcast1dVectorFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualSimpleUint8,            NotEqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1ElementUint8, NotEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1dVectorUint8, NotEqualBroadcast1dVectorUint8Test)

// Max
ARMNN_AUTO_TEST_CASE(SimpleMaximum, MaximumSimpleTest)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1Element, MaximumBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVector, MaximumBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE(MaximumFloat16, MaximumFloat16Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1ElementFloat16, MaximumBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVectorFloat16, MaximumBroadcast1DVectorFloat16Test)
ARMNN_AUTO_TEST_CASE(MaximumUint8, MaximumUint8Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1ElementUint8, MaximumBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVectorUint8, MaximumBroadcast1DVectorUint8Test)
ARMNN_AUTO_TEST_CASE(MaximumInt16, MaximumInt16Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1ElementInt16, MaximumBroadcast1ElementInt16Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVectorInt16, MaximumBroadcast1DVectorInt16Test)
ARMNN_AUTO_TEST_CASE(MaximumInt32, MaximumInt32Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1ElementInt32, MaximumBroadcast1ElementInt32Test)
ARMNN_AUTO_TEST_CASE(MaximumBroadcast1DVectorInt32, MaximumBroadcast1DVectorInt32Test)

// Min
ARMNN_AUTO_TEST_CASE(SimpleMinimum1, MinimumBroadcast1ElementTest1)
ARMNN_AUTO_TEST_CASE(SimpleMinimum2, MinimumBroadcast1ElementTest2)
ARMNN_AUTO_TEST_CASE(Minimum1DVectorUint8, MinimumBroadcast1DVectorUint8Test)
ARMNN_AUTO_TEST_CASE(MinimumFloat16, MinimumFloat16Test)
ARMNN_AUTO_TEST_CASE(MinimumBroadcast1ElementFloat16, MinimumBroadcast1ElementFloat16Test)
ARMNN_AUTO_TEST_CASE(MinimumBroadcast1DVectorFloat16, MinimumBroadcast1DVectorFloat16Test)
ARMNN_AUTO_TEST_CASE(MinimumInt16, MinimumInt16Test)
ARMNN_AUTO_TEST_CASE(MinimumBroadcast1ElementInt16, MinimumBroadcast1ElementInt16Test)
ARMNN_AUTO_TEST_CASE(MinimumBroadcast1DVectorInt16, MinimumBroadcast1DVectorInt16Test)
ARMNN_AUTO_TEST_CASE(MinimumInt32, MinimumInt32Test)
ARMNN_AUTO_TEST_CASE(MinimumBroadcast1ElementInt32, MinimumBroadcast1ElementInt32Test)
ARMNN_AUTO_TEST_CASE(MinimumBroadcast1DVectorInt32, MinimumBroadcast1DVectorInt32Test)

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
ARMNN_AUTO_TEST_CASE(MultiplicationInt32, MultiplicationInt32Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1ElementInt32, MultiplicationBroadcast1ElementInt32Test)
ARMNN_AUTO_TEST_CASE(MultiplicationBroadcast1DVectorInt32, MultiplicationBroadcast1DVectorInt32Test)
ARMNN_AUTO_TEST_CASE(Multiplication5d, Multiplication5dTest)

// Batch Norm
ARMNN_AUTO_TEST_CASE(BatchNormFloat32, BatchNormFloat32Test)
ARMNN_AUTO_TEST_CASE(BatchNormFloat32Nhwc, BatchNormFloat32NhwcTest)
ARMNN_AUTO_TEST_CASE(BatchNormFloat16, BatchNormFloat16Test)
ARMNN_AUTO_TEST_CASE(BatchNormFloat16Nhwc, BatchNormFloat16NhwcTest)
ARMNN_AUTO_TEST_CASE(BatchNormUint8, BatchNormUint8Test)
ARMNN_AUTO_TEST_CASE(BatchNormUint8Nhwc, BatchNormUint8NhwcTest)
ARMNN_AUTO_TEST_CASE(BatchNormInt16, BatchNormInt16Test)
ARMNN_AUTO_TEST_CASE(BatchNormInt16Nhwc, BatchNormInt16NhwcTest)

// Rank
ARMNN_AUTO_TEST_CASE(RankDimSize1Float16,  RankDimSize1Test<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(RankDimSize1Float32,  RankDimSize1Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(RankDimSize1QAsymmU8, RankDimSize1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(RankDimSize1Signed32, RankDimSize1Test<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE(RankDimSize1QSymmS16, RankDimSize1Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(RankDimSize1QSymmS8,  RankDimSize1Test<DataType::QSymmS8>)
ARMNN_AUTO_TEST_CASE(RankDimSize1QAsymmS8, RankDimSize1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(RankDimSize1BFloat16, RankDimSize1Test<DataType::BFloat16>)

ARMNN_AUTO_TEST_CASE(RankDimSize2Float16,  RankDimSize2Test<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(RankDimSize2Float32,  RankDimSize2Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(RankDimSize2QAsymmU8, RankDimSize2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(RankDimSize2Signed32, RankDimSize2Test<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE(RankDimSize2QSymmS16, RankDimSize2Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(RankDimSize2QSymmS8,  RankDimSize2Test<DataType::QSymmS8>)
ARMNN_AUTO_TEST_CASE(RankDimSize2QAsymmS8, RankDimSize2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(RankDimSize2BFloat16, RankDimSize2Test<DataType::BFloat16>)

ARMNN_AUTO_TEST_CASE(RankDimSize3Float16,  RankDimSize3Test<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(RankDimSize3Float32,  RankDimSize3Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(RankDimSize3QAsymmU8, RankDimSize3Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(RankDimSize3Signed32, RankDimSize3Test<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE(RankDimSize3QSymmS16, RankDimSize3Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(RankDimSize3QSymmS8,  RankDimSize3Test<DataType::QSymmS8>)
ARMNN_AUTO_TEST_CASE(RankDimSize3QAsymmS8, RankDimSize3Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(RankDimSize3BFloat16, RankDimSize3Test<DataType::BFloat16>)

ARMNN_AUTO_TEST_CASE(RankDimSize4Float16,  RankDimSize4Test<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(RankDimSize4Float32,  RankDimSize4Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(RankDimSize4QAsymmU8, RankDimSize4Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(RankDimSize4Signed32, RankDimSize4Test<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE(RankDimSize4QSymmS16, RankDimSize4Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(RankDimSize4QSymmS8,  RankDimSize4Test<DataType::QSymmS8>)
ARMNN_AUTO_TEST_CASE(RankDimSize4QAsymmS8, RankDimSize4Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(RankDimSize4BFloat16, RankDimSize4Test<DataType::BFloat16>)

// Resize Bilinear - NCHW
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinear,
                     SimpleResizeBilinearTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearFloat16,
                     SimpleResizeBilinearTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearInt8,
                     SimpleResizeBilinearTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearUint8,
                     SimpleResizeBilinearTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearUint16,
                     SimpleResizeBilinearTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNop,
                     ResizeBilinearNopTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopFloat16,
                     ResizeBilinearNopTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopInt8,
                     ResizeBilinearNopTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopUint8,
                     ResizeBilinearNopTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(esizeBilinearNopUint16,
                     SimpleResizeBilinearTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMin,
                     ResizeBilinearSqMinTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinFloat16,
                     ResizeBilinearSqMinTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinInt8,
                     ResizeBilinearSqMinTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinUint8,
                     ResizeBilinearSqMinTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinUint16,
                     SimpleResizeBilinearTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMin,
                     ResizeBilinearMinTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinFloat16,
                     ResizeBilinearMinTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinInt8,
                     ResizeBilinearMinTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinUint8,
                     ResizeBilinearMinTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinUint16,
                     SimpleResizeBilinearTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMag,
                     ResizeBilinearMagTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagFloat16,
                     ResizeBilinearMagTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagInt8,
                     ResizeBilinearMagTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagUint8,
                     ResizeBilinearMagTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagUint16,
                     SimpleResizeBilinearTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinear,
                     HalfPixelCentersResizeBilinearTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinearFloat16,
                     HalfPixelCentersResizeBilinearTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinearInt8,
                     HalfPixelCentersResizeBilinearTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinearUint8,
                     HalfPixelCentersResizeBilinearTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinearUint16,
                     HalfPixelCentersResizeBilinearTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinear,
                     AlignCornersResizeBilinearTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinearFloat16,
                     AlignCornersResizeBilinearTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinearInt8,
                     AlignCornersResizeBilinearTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinearUint8,
                     AlignCornersResizeBilinearTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinearUint16,
                     AlignCornersResizeBilinearTest<DataType::QSymmS16>,
                     DataLayout::NCHW)

// Resize Bilinear - NHWC
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopNhwc,
                     ResizeBilinearNopTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopNhwcFloat16,
                     ResizeBilinearNopTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopInt8Nhwc,
                     ResizeBilinearNopTest<DataType::QAsymmS8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopUint8Nhwc,
                     ResizeBilinearNopTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearNopUint16Nhwc,
                     ResizeBilinearNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearNhwc,
                     SimpleResizeBilinearTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearNhwcFloat16,
                     SimpleResizeBilinearTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearInt8Nhwc,
                     SimpleResizeBilinearTest<DataType::QAsymmS8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearUint8Nhwc,
                     SimpleResizeBilinearTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeBilinearUint16Nhwc,
                     ResizeBilinearNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinNhwc,
                     ResizeBilinearSqMinTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinNhwcFloat16,
                     ResizeBilinearSqMinTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinInt8Nhwc,
                     ResizeBilinearSqMinTest<DataType::QAsymmS8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinUint8Nhwc,
                     ResizeBilinearSqMinTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearSqMinUint16Nhwc,
                     ResizeBilinearNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinNhwc,
                     ResizeBilinearMinTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinNhwcFloat16,
                     ResizeBilinearMinTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinInt8Nhwc,
                     ResizeBilinearMinTest<DataType::QAsymmS8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinUint8Nhwc,
                     ResizeBilinearMinTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMinUint16Nhwc,
                     ResizeBilinearNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagNhwc,
                     ResizeBilinearMagTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagNhwcFloat16,
                     ResizeBilinearMagTest<DataType::Float16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagInt8Nhwc,
                     ResizeBilinearMagTest<DataType::QAsymmS8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagUint8Nhwc,
                     ResizeBilinearMagTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeBilinearMagUint16Nhwc,
                     ResizeBilinearNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinearNhwc,
                     HalfPixelCentersResizeBilinearTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinearFloat16Nhwc,
                     HalfPixelCentersResizeBilinearTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinearInt8Nhwc,
                     HalfPixelCentersResizeBilinearTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinearUint8Nhwc,
                     HalfPixelCentersResizeBilinearTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeBilinearUint16Nhwc,
                     HalfPixelCentersResizeBilinearTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinearNhwc,
                     AlignCornersResizeBilinearTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinearFloat16Nhwc,
                     AlignCornersResizeBilinearTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinearInt8Nhwc,
                     AlignCornersResizeBilinearTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinearUint8Nhwc,
                     AlignCornersResizeBilinearTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeBilinearUint16Nhwc,
                     AlignCornersResizeBilinearTest<DataType::QSymmS16>,
                     DataLayout::NCHW)

// Resize NearestNeighbor - NCHW
ARMNN_AUTO_TEST_CASE(SimpleResizeNearestNeighbor,
                     SimpleResizeNearestNeighborTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleResizeNearestNeighborInt8,
                     SimpleResizeNearestNeighborTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleResizeNearestNeighborUint8,
                     SimpleResizeNearestNeighborTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleResizeNearestNeighborUint16,
                     SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborNop,
                     ResizeNearestNeighborNopTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborNopInt8,
                     ResizeNearestNeighborNopTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborNopUint8,
                     ResizeNearestNeighborNopTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(esizeNearestNeighborNopUint16,
                     SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborSqMin,
                     ResizeNearestNeighborSqMinTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborSqMinInt8,
                     ResizeNearestNeighborSqMinTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborSqMinUint8,
                     ResizeNearestNeighborSqMinTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborSqMinUint16,
                     SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMin,
                     ResizeNearestNeighborMinTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMinInt8,
                     ResizeNearestNeighborMinTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMinUint8,
                     ResizeNearestNeighborMinTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMinUint16,
                     SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMag,
                     ResizeNearestNeighborMagTest<DataType::Float32>,
                     DataLayout::NCHW, 0.10f, 50, 0.11f, 20)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMagInt8,
                     ResizeNearestNeighborMagTest<DataType::QAsymmS8>,
                     DataLayout::NCHW, 0.10f, 50, 0.11f, 20)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMagUint8,
                     ResizeNearestNeighborMagTest<DataType::QAsymmU8>,
                     DataLayout::NCHW, 0.10f, 50, 0.11f, 20)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMagUint16,
                     SimpleResizeNearestNeighborTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbour,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbourFloat16,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbourInt8,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbourUint8,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbourUint16,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbour,
                     AlignCornersResizeNearestNeighbourTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbourFloat16,
                     AlignCornersResizeNearestNeighbourTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbourInt8,
                     AlignCornersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbourUint8,
                     AlignCornersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbourUint16,
                     AlignCornersResizeNearestNeighbourTest<DataType::QSymmS16>,
                     DataLayout::NCHW)

// Resize NearestNeighbor - NHWC
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborNopNhwc,
                     ResizeNearestNeighborNopTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborNopInt8Nhwc,
                     ResizeNearestNeighborNopTest<DataType::QAsymmS8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborNopUint8Nhwc,
                     ResizeNearestNeighborNopTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborNopUint16Nhwc,
                     ResizeNearestNeighborNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeNearestNeighborNhwc,
                     SimpleResizeNearestNeighborTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeNearestNeighborInt8Nhwc,
                     SimpleResizeNearestNeighborTest<DataType::QAsymmS8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeNearestNeighborUint8Nhwc,
                     SimpleResizeNearestNeighborTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleResizeNearestNeighborUint16Nhwc,
                     ResizeNearestNeighborNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborSqMinNhwc,
                     ResizeNearestNeighborSqMinTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborSqMinInt8Nhwc,
                     ResizeNearestNeighborSqMinTest<DataType::QAsymmS8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborSqMinUint8Nhwc,
                     ResizeNearestNeighborSqMinTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborSqMinUint16Nhwc,
                     ResizeNearestNeighborNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMinNhwc,
                     ResizeNearestNeighborMinTest<DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMinInt8Nhwc,
                     ResizeNearestNeighborMinTest<DataType::QAsymmS8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMinUint8Nhwc,
                     ResizeNearestNeighborMinTest<DataType::QAsymmU8>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMinUint16Nhwc,
                     ResizeNearestNeighborNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMagNhwc,
                     ResizeNearestNeighborMagTest<DataType::Float32>,
                     DataLayout::NHWC, 0.10f, 50, 0.11f, 20)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMagInt8Nhwc,
                     ResizeNearestNeighborMagTest<DataType::QAsymmS8>,
                     DataLayout::NHWC, 0.10f, 50, 0.11f, 20)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMagUint8Nhwc,
                     ResizeNearestNeighborMagTest<DataType::QAsymmU8>,
                     DataLayout::NHWC, 0.10f, 50, 0.11f, 20)
ARMNN_AUTO_TEST_CASE(ResizeNearestNeighborMagUint16Nhwc,
                     ResizeNearestNeighborNopTest<DataType::QSymmS16>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbourNchw,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbourFloat16Nchw,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbourInt8Nchw,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbourUint8Nchw,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(HalfPixelCentersResizeNearestNeighbourUint16Nchw,
                     HalfPixelCentersResizeNearestNeighbourTest<DataType::QSymmS16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbourNchw,
                     AlignCornersResizeNearestNeighbourTest<DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbourFloat16Nchw,
                     AlignCornersResizeNearestNeighbourTest<DataType::Float16>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbourInt8Nchw,
                     AlignCornersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbourUint8Nchw,
                     AlignCornersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(AlignCornersResizeNearestNeighbourUint16Nchw,
                     AlignCornersResizeNearestNeighbourTest<DataType::QSymmS16>,
                     DataLayout::NCHW)

// Fake Quantization
ARMNN_AUTO_TEST_CASE_WITH_THF(FakeQuantization, FakeQuantizationTest)

// L2 Normalization
ARMNN_AUTO_TEST_CASE(L2Normalization1d, L2Normalization1dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization2d, L2Normalization2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization3d, L2Normalization3dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization4d, L2Normalization4dTest, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(L2Normalization1dInt16, L2Normalization1dInt16Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization2dInt16, L2Normalization2dInt16Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization3dInt16, L2Normalization3dInt16Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization4dInt16, L2Normalization4dInt16Test, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(L2Normalization1dUint8, L2Normalization1dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization2dUint8, L2Normalization2dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization3dUint8, L2Normalization3dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2Normalization4dUint8, L2Normalization4dUint8Test, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(L2Normalization1dNhwc, L2Normalization1dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization2dNhwc, L2Normalization2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization3dNhwc, L2Normalization3dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization4dNhwc, L2Normalization4dTest, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(L2Normalization1dInt16Nhwc, L2Normalization1dInt16Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization2dInt16Nhwc, L2Normalization2dInt16Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization3dInt16Nhwc, L2Normalization3dInt16Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization4dInt16Nhwc, L2Normalization4dInt16Test, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(L2Normalization1dUint8Nhwc, L2Normalization1dUint8Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization2dUint8Nhwc, L2Normalization2dUint8Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization3dUint8Nhwc, L2Normalization3dUint8Test, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(L2Normalization4dUint8Nhwc, L2Normalization4dUint8Test, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE(L2Normalization2dShape, L2Normalization2dShapeTest);

ARMNN_AUTO_TEST_CASE(L2NormalizationDefaultEpsilon, L2NormalizationDefaultEpsilonTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(L2NormalizationNonDefaultEpsilon, L2NormalizationNonDefaultEpsilonTest, DataLayout::NCHW)

// LogSoftmax
ARMNN_AUTO_TEST_CASE(LogSoftmaxFloat32_1, LogSoftmaxTest1<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(LogSoftmaxFloat32_2, LogSoftmaxTest2<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(LogSoftmaxFloat32_3, LogSoftmaxTest3<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(LogSoftmaxFloat32_4, LogSoftmaxTest4<DataType::Float32>)

ARMNN_AUTO_TEST_CASE(LogSoftmaxFloat16_1, LogSoftmaxTest1<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(LogSoftmaxFloat16_2, LogSoftmaxTest2<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(LogSoftmaxFloat16_3, LogSoftmaxTest3<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(LogSoftmaxFloat16_4, LogSoftmaxTest4<DataType::Float16>)

// Pad
ARMNN_AUTO_TEST_CASE(PadBFloat162d, PadBFloat162dTest)
ARMNN_AUTO_TEST_CASE(PadBFloat162dCustomPadding, PadBFloat162dCustomPaddingTest)
ARMNN_AUTO_TEST_CASE(PadBFloat163d, PadBFloat163dTest)
ARMNN_AUTO_TEST_CASE(PadBFloat164d, PadBFloat164dTest)

ARMNN_AUTO_TEST_CASE(PadFloat322d, PadFloat322dTest)
ARMNN_AUTO_TEST_CASE(PadFloat322dCustomPadding, PadFloat322dCustomPaddingTest)
ARMNN_AUTO_TEST_CASE(PadFloat323d, PadFloat323dTest)
ARMNN_AUTO_TEST_CASE(PadFloat324d, PadFloat324dTest)

ARMNN_AUTO_TEST_CASE(PadUint82d, PadUint82dTest)
ARMNN_AUTO_TEST_CASE(PadUint82dCustomPadding, PadUint82dCustomPaddingTest)
ARMNN_AUTO_TEST_CASE(PadUint83d, PadUint83dTest)
ARMNN_AUTO_TEST_CASE(PadUint84d, PadUint84dTest)

ARMNN_AUTO_TEST_CASE(Pad2dQSymm16, Pad2dTestCommon<DataType::QSymmS16>, 2.0f, 0, 0.0f)
ARMNN_AUTO_TEST_CASE(Pad2dQSymm16CustomPadding, Pad2dTestCommon<DataType::QSymmS16>, 2.0f, 0, 1.0f)
ARMNN_AUTO_TEST_CASE(Pad3dQSymm16, Pad3dTestCommon<DataType::QSymmS16>, 2.0f, 0)
ARMNN_AUTO_TEST_CASE(Pad4dQSymm16, Pad4dTestCommon<DataType::QSymmS16>, 2.0f, 0)

ARMNN_AUTO_TEST_CASE(PadInt82d, PadInt82dTest)
ARMNN_AUTO_TEST_CASE(PadInt82dCustomPadding, PadInt82dCustomPaddingTest)
ARMNN_AUTO_TEST_CASE(PadInt83d, PadInt83dTest)
ARMNN_AUTO_TEST_CASE(PadInt84d, PadInt84dTest)

// Constant
ARMNN_AUTO_TEST_CASE_WITH_THF(Constant, ConstantTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConstantUint8, ConstantUint8CustomQuantizationScaleAndOffsetTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConstantInt16, ConstantInt16CustomQuantizationScaleAndOffsetTest)

// Concat
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat1d, Concat1dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat1dUint8, Concat1dUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim0, Concat2dDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim0Uint8, Concat2dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim1, Concat2dDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim1Uint8, Concat2dDim1Uint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim0DiffInputDims, Concat2dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim0DiffInputDimsUint8, Concat2dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim1DiffInputDims, Concat2dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat2dDim1DiffInputDimsUint8, Concat2dDim1DiffInputDimsUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0, Concat3dDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0Uint8, Concat3dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1, Concat3dDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1Uint8, Concat3dDim1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2, Concat3dDim2Test, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2Uint8, Concat3dDim2Uint8Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0DiffInputDims, Concat3dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0DiffInputDimsUint8, Concat3dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1DiffInputDims, Concat3dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1DiffInputDimsUint8, Concat3dDim1DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2DiffInputDims, Concat3dDim2DiffInputDimsTest, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2DiffInputDimsUint8, Concat3dDim2DiffInputDimsUint8Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim0, Concat4dDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim1, Concat4dDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim2, Concat4dDim2Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim3, Concat4dDim3Test, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim0Uint8, Concat4dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim1Uint8, Concat4dDim1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim2Uint8, Concat4dDim2Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim3Uint8, Concat4dDim3Uint8Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim0, Concat4dDiffShapeDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim1, Concat4dDiffShapeDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim2, Concat4dDiffShapeDim2Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim3, Concat4dDiffShapeDim3Test, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim0Uint8, Concat4dDiffShapeDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim1Uint8, Concat4dDiffShapeDim1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim2Uint8, Concat4dDiffShapeDim2Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim3Uint8, Concat4dDiffShapeDim3Uint8Test, true)

// Fill
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFill, SimpleFillTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFillF16, SimpleFillTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFillS32, SimpleFillTest<DataType::Signed32>)

// Floor
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFloor, SimpleFloorTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFloorFloat16, SimpleFloorTest<DataType::Float16>)

// Reshape
ARMNN_AUTO_TEST_CASE(SimpleReshapeFloat32, SimpleReshapeTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(SimpleReshapeQuantisedAsymmS8, SimpleReshapeTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(SimpleReshapeQuantisedAsymm8, SimpleReshapeTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(SimpleReshapeQuantisedSymm16, SimpleReshapeTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(Reshape5d, Reshape5dTest<DataType::Float32>)

// Rsqrt
ARMNN_AUTO_TEST_CASE(Rsqrt2d, Rsqrt2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Rsqrt3d, Rsqrt3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(RsqrtZero, RsqrtZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(RsqrtNegative, RsqrtNegativeTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Rsqrt2dFloat16, Rsqrt2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(Rsqrt3dFloat16, Rsqrt3dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(Rsqrt2dQuantisedAsymmS8, Rsqrt2dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(Rsqrt3dQuantisedAsymmS8, Rsqrt3dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(Rsqrt2dQuantisedAsymm8, Rsqrt2dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(Rsqrt3dQuantisedAsymm8, Rsqrt3dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(Rsqrt2dQuantisedSymm16, Rsqrt2dTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(Rsqrt3dQuantisedSymm16, Rsqrt3dTest<DataType::QSymmS16>)

// Permute
ARMNN_AUTO_TEST_CASE(SimplePermuteBFloat16, SimplePermuteTest<DataType::BFloat16>)
ARMNN_AUTO_TEST_CASE(PermuteBFloat16ValueSet1Test, PermuteValueSet1Test<DataType::BFloat16>)
ARMNN_AUTO_TEST_CASE(PermuteBFloat16ValueSet2Test, PermuteValueSet2Test<DataType::BFloat16>)
ARMNN_AUTO_TEST_CASE(PermuteBFloat16ValueSet3Test, PermuteValueSet3Test<DataType::BFloat16>)
ARMNN_AUTO_TEST_CASE(SimplePermuteFloat32, SimplePermuteTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(PermuteFloat32ValueSet1Test, PermuteValueSet1Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(PermuteFloat32ValueSet2Test, PermuteValueSet2Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(PermuteFloat32ValueSet3Test, PermuteValueSet3Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(SimplePermuteQASymS8, SimplePermuteTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(PermuteQASymmS8ValueSet1Test, PermuteValueSet1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(PermuteQASymmS8ValueSet2Test, PermuteValueSet2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(PermuteQASymmS8ValueSet3Test, PermuteValueSet3Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(SimplePermuteQASymm8, SimplePermuteTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(PermuteQASymm8ValueSet1Test, PermuteValueSet1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(PermuteQASymm8ValueSet2Test, PermuteValueSet2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(PermuteQASymm8ValueSet3Test, PermuteValueSet3Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(SimplePermuteQSymm16, SimplePermuteTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(PermuteQSymm16ValueSet1Test, PermuteValueSet1Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(PermuteQSymm16ValueSet2Test, PermuteValueSet2Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(PermuteQSymm16ValueSet3Test, PermuteValueSet3Test<DataType::QSymmS16>)

// Lstm
BOOST_AUTO_TEST_CASE(LstmUtilsZeroVector) {
                     LstmUtilsZeroVectorTest(); }
BOOST_AUTO_TEST_CASE(LstmUtilsMeanStddevNormalization) {
                     LstmUtilsMeanStddevNormalizationNoneZeroInputTest();
                     LstmUtilsMeanStddevNormalizationAllZeroInputTest();
                     LstmUtilsMeanStddevNormalizationMixedZeroInputTest(); }
BOOST_AUTO_TEST_CASE(LstmUtilsVectorBatchVectorCwiseProduct) {
                     LstmUtilsVectorBatchVectorCwiseProductTest(); }
BOOST_AUTO_TEST_CASE(LstmUtilsVectorBatchVectorAdd) {
                     LstmUtilsVectorBatchVectorAddTest(); }

ARMNN_AUTO_TEST_CASE(LstmLayerFloat32WithCifgWithPeepholeNoProjection,
                     LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerFloat32NoCifgNoPeepholeNoProjection,
                     LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerFloat32NoCifgWithPeepholeWithProjection,
                     LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest)

ARMNN_AUTO_TEST_CASE(LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNorm,
                     LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNormTest)

ARMNN_AUTO_TEST_CASE(LstmLayerInt16NoCifgNoPeepholeNoProjection,
                     LstmLayerInt16NoCifgNoPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerInt16WithCifgWithPeepholeNoProjection,
                     LstmLayerInt16WithCifgWithPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerInt16NoCifgWithPeepholeWithProjection,
                     LstmLayerInt16NoCifgWithPeepholeWithProjectionTest)
ARMNN_AUTO_TEST_CASE(LstmLayerInt16NoCifgNoPeepholeNoProjectionInt16Constant,
                     LstmLayerInt16NoCifgNoPeepholeNoProjectionInt16ConstantTest)

// QLstm
ARMNN_AUTO_TEST_CASE(QLstm, QLstmTest)
ARMNN_AUTO_TEST_CASE(QLstm1, QLstmTest1)
ARMNN_AUTO_TEST_CASE(QLstm2, QLstmTest2)

// Convert from BFloat16 to Float32
ARMNN_AUTO_TEST_CASE_WITH_THF(ConvertBf16ToFp32, ConvertBf16ToFp32Test)

// Convert from Float32 to BFloat16
ARMNN_AUTO_TEST_CASE_WITH_THF(ConvertFp32ToBf16, ConvertFp32ToBf16Test)

// Convert from Float16 to Float32
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvertFp16ToFp32, SimpleConvertFp16ToFp32Test)
// Convert from Float32 to Float16
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvertFp32ToFp16, SimpleConvertFp32ToFp16Test)

// Mean
ARMNN_AUTO_TEST_CASE(MeanSimpleFloat32, MeanSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(MeanSimpleAxisFloat32, MeanSimpleAxisTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(MeanKeepDimsFloat32, MeanKeepDimsTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(MeanMultipleDimsFloat32, MeanMultipleDimsTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(MeanVts1Float32, MeanVts1Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(MeanVts2Float32, MeanVts2Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(MeanVts3Float32, MeanVts3Test<DataType::Float32>)

ARMNN_AUTO_TEST_CASE(MeanSimpleQuantisedAsymmS8, MeanSimpleTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(MeanSimpleAxisQuantisedAsymmS8, MeanSimpleAxisTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(MeanKeepDimsQuantisedAsymmS8, MeanKeepDimsTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(MeanMultipleDimsQuantisedAsymmS8, MeanMultipleDimsTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(MeanVts1QuantisedAsymmS8, MeanVts1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(MeanVts2QuantisedAsymmS8, MeanVts2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(MeanVts3QuantisedAsymmS8, MeanVts3Test<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_CASE(MeanSimpleQuantisedAsymm8, MeanSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(MeanSimpleAxisQuantisedAsymm8, MeanSimpleAxisTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(MeanKeepDimsQuantisedAsymm8, MeanKeepDimsTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(MeanMultipleDimsQuantisedAsymm8, MeanMultipleDimsTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(MeanVts1QuantisedAsymm8, MeanVts1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(MeanVts2QuantisedAsymm8, MeanVts2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(MeanVts3QuantisedAsymm8, MeanVts3Test<DataType::QAsymmU8>)

ARMNN_AUTO_TEST_CASE(MeanSimpleQuantisedSymm16, MeanSimpleTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(MeanSimpleAxisQuantisedSymm16, MeanSimpleAxisTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(MeanKeepDimsQuantisedSymm16, MeanKeepDimsTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(MeanMultipleDimsQuantisedSymm16, MeanMultipleDimsTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(MeanVts1QuantisedSymm16, MeanVts1Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(MeanVts2QuantisedSymm16, MeanVts2Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(MeanVts3QuantisedSymm16, MeanVts3Test<DataType::QSymmS16>)

ARMNN_AUTO_TEST_CASE(AdditionAfterMaxPool, AdditionAfterMaxPoolTest)

// ArgMinMax
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxFloat32, ArgMaxSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinFloat32, ArgMinSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelFloat32, ArgMinChannelTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelFloat32, ArgMaxChannelTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxHeightFloat32, ArgMaxHeightTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinWidthFloat32, ArgMinWidthTest<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxFloat16, ArgMaxSimpleTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinFloat16, ArgMinSimpleTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelFloat16, ArgMinChannelTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelFloat16, ArgMaxChannelTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxHeightFloat16, ArgMaxHeightTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinWidthFloat16, ArgMinWidthTest<DataType::Float16>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxSigned32, ArgMaxSimpleTest<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinSigned32, ArgMinSimpleTest<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelSigned32, ArgMinChannelTest<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelSigned32, ArgMaxChannelTest<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxHeightSigned32, ArgMaxHeightTest<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinWidthSigned32, ArgMinWidthTest<DataType::Signed32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxSimpleQuantisedAsymmS8, ArgMaxSimpleTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinSimpleQuantisedAsymmS8, ArgMinSimpleTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelQuantisedAsymmS8, ArgMinChannelTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelQuantisedAsymmS8, ArgMaxChannelTest<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxSimpleQuantisedAsymm8, ArgMaxSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinSimpleQuantisedAsymm8, ArgMinSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelQuantisedAsymm8, ArgMinChannelTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelQuantisedAsymm8, ArgMaxChannelTest<DataType::QAsymmU8>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxSimpleQuantisedSymm16, ArgMaxSimpleTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinSimpleQuantisedSymm16, ArgMinSimpleTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelQuantisedSymm16, ArgMinChannelTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelQuantisedSymm16, ArgMaxChannelTest<DataType::QSymmS16>)

// Space To Batch Nd
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleFloat32, SpaceToBatchNdSimpleFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsFloat32, SpaceToBatchNdMultiChannelsFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockFloat32, SpaceToBatchNdMultiBlockFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingFloat32, SpaceToBatchNdPaddingFloat32Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleFloat16, SpaceToBatchNdSimpleFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsFloat16, SpaceToBatchNdMultiChannelsFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockFloat16, SpaceToBatchNdMultiBlockFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingFloat16, SpaceToBatchNdPaddingFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleUint8, SpaceToBatchNdSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsUint8, SpaceToBatchNdMultiChannelsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockUint8, SpaceToBatchNdMultiBlockUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingUint8, SpaceToBatchNdPaddingUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcFloat32, SpaceToBatchNdSimpleNhwcFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcFloat32, SpaceToBatchNdMultiChannelsNhwcFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcFloat32, SpaceToBatchNdMultiBlockNhwcFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcFloat32, SpaceToBatchNdPaddingNhwcFloat32Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcFloat16, SpaceToBatchNdSimpleNhwcFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcFloat16, SpaceToBatchNdMultiChannelsNhwcFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcFloat16, SpaceToBatchNdMultiBlockNhwcFloat16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcFloat16, SpaceToBatchNdPaddingNhwcFloat16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcUint8, SpaceToBatchNdSimpleNhwcUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcUint8, SpaceToBatchNdMultiChannelsNhwcUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcUint8, SpaceToBatchNdMultiBlockNhwcUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcUint8, SpaceToBatchNdPaddingNhwcUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleUint16, SpaceToBatchNdSimpleUint16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsUint16, SpaceToBatchNdMultiChannelsUint16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockUint16, SpaceToBatchNdMultiBlockUint16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingUint16, SpaceToBatchNdPaddingUint16Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcUint16, SpaceToBatchNdSimpleNhwcUint16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcUint16, SpaceToBatchNdMultiChannelsNhwcUint16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcUint16, SpaceToBatchNdMultiBlockNhwcUint16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcUint16, SpaceToBatchNdPaddingNhwcUint16Test)

// BatchToSpace
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat32_1, BatchToSpaceNdNhwcTest1<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat32_2, BatchToSpaceNdNhwcTest2<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat32_3, BatchToSpaceNdNhwcTest3<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat32_4, BatchToSpaceNdNhwcTest4<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat32_5, BatchToSpaceNdNhwcTest5<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat32_6, BatchToSpaceNdNhwcTest6<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat32_7, BatchToSpaceNdNhwcTest7<DataType::Float32>)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat16_1, BatchToSpaceNdNhwcTest1<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat16_2, BatchToSpaceNdNhwcTest2<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat16_3, BatchToSpaceNdNhwcTest3<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat16_4, BatchToSpaceNdNhwcTest4<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat16_5, BatchToSpaceNdNhwcTest5<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat16_6, BatchToSpaceNdNhwcTest6<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcFloat16_7, BatchToSpaceNdNhwcTest7<DataType::Float16>)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcInt1,  BatchToSpaceNdNhwcTest1<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcInt2,  BatchToSpaceNdNhwcTest2<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcInt3,  BatchToSpaceNdNhwcTest3<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcInt4,  BatchToSpaceNdNhwcTest4<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcInt5,  BatchToSpaceNdNhwcTest5<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcInt6,  BatchToSpaceNdNhwcTest6<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcInt7,  BatchToSpaceNdNhwcTest7<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint1,  BatchToSpaceNdNhwcTest1<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint2,  BatchToSpaceNdNhwcTest2<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint3,  BatchToSpaceNdNhwcTest3<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint4,  BatchToSpaceNdNhwcTest4<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint5,  BatchToSpaceNdNhwcTest5<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint6,  BatchToSpaceNdNhwcTest6<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcUint7,  BatchToSpaceNdNhwcTest7<DataType::QAsymmU8>)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcQsymm16_1,  BatchToSpaceNdNhwcTest1<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcQsymm16_2,  BatchToSpaceNdNhwcTest2<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcQsymm16_3,  BatchToSpaceNdNhwcTest3<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcQsymm16_4,  BatchToSpaceNdNhwcTest4<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcQsymm16_5,  BatchToSpaceNdNhwcTest5<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcQsymm16_6,  BatchToSpaceNdNhwcTest6<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNhwcQsymm16_7,  BatchToSpaceNdNhwcTest7<DataType::QSymmS16>)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat16_1, BatchToSpaceNdNchwTest1<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat16_2, BatchToSpaceNdNchwTest2<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat16_3, BatchToSpaceNdNchwTest3<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat16_4, BatchToSpaceNdNchwTest4<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat16_5, BatchToSpaceNdNchwTest5<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat16_6, BatchToSpaceNdNchwTest6<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwFloat16_7, BatchToSpaceNdNchwTest7<DataType::Float16>)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwInt1,  BatchToSpaceNdNchwTest1<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwInt2,  BatchToSpaceNdNchwTest2<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwInt3,  BatchToSpaceNdNchwTest3<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwInt4,  BatchToSpaceNdNchwTest4<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwInt5,  BatchToSpaceNdNchwTest5<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwInt6,  BatchToSpaceNdNchwTest6<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwInt7,  BatchToSpaceNdNchwTest7<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint1,  BatchToSpaceNdNchwTest1<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint2,  BatchToSpaceNdNchwTest2<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint3,  BatchToSpaceNdNchwTest3<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint4,  BatchToSpaceNdNchwTest4<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint5,  BatchToSpaceNdNchwTest5<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint6,  BatchToSpaceNdNchwTest6<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwUint7,  BatchToSpaceNdNchwTest7<DataType::QAsymmU8>)

ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwQsymm16_1,  BatchToSpaceNdNchwTest1<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwQsymm16_2,  BatchToSpaceNdNchwTest2<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwQsymm16_3,  BatchToSpaceNdNchwTest3<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwQsymm16_4,  BatchToSpaceNdNchwTest4<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwQsymm16_5,  BatchToSpaceNdNchwTest5<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwQsymm16_6,  BatchToSpaceNdNchwTest6<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(BatchToSpaceNdNchwQsymm16_7,  BatchToSpaceNdNchwTest7<DataType::QSymmS16>)

// DepthToSpace
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat32_1, DepthToSpaceTest1<DataType::Float32>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat32_2, DepthToSpaceTest2<DataType::Float32>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat32_3, DepthToSpaceTest3<DataType::Float32>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat32_4, DepthToSpaceTest4<DataType::Float32>, DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat16_1, DepthToSpaceTest1<DataType::Float16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat16_2, DepthToSpaceTest2<DataType::Float16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat16_3, DepthToSpaceTest3<DataType::Float16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwFloat16_4, DepthToSpaceTest4<DataType::Float16>, DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt8_1, DepthToSpaceTest1<DataType::QAsymmS8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt8_2, DepthToSpaceTest2<DataType::QAsymmS8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt8_3, DepthToSpaceTest3<DataType::QAsymmS8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt8_4, DepthToSpaceTest4<DataType::QAsymmS8>, DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwUint8_1, DepthToSpaceTest1<DataType::QAsymmU8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwUint8_2, DepthToSpaceTest2<DataType::QAsymmU8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwUint8_3, DepthToSpaceTest3<DataType::QAsymmU8>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwUint8_4, DepthToSpaceTest4<DataType::QAsymmU8>, DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt16_1, DepthToSpaceTest1<DataType::QSymmS16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt16_2, DepthToSpaceTest2<DataType::QSymmS16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt16_3, DepthToSpaceTest3<DataType::QSymmS16>, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNchwInt16_4, DepthToSpaceTest4<DataType::QSymmS16>, DataLayout::NCHW);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat32_1, DepthToSpaceTest1<DataType::Float32>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat32_2, DepthToSpaceTest2<DataType::Float32>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat32_3, DepthToSpaceTest3<DataType::Float32>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat32_4, DepthToSpaceTest4<DataType::Float32>, DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat16_1, DepthToSpaceTest1<DataType::Float16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat16_2, DepthToSpaceTest2<DataType::Float16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat16_3, DepthToSpaceTest3<DataType::Float16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcFloat16_4, DepthToSpaceTest4<DataType::Float16>, DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt8_1, DepthToSpaceTest1<DataType::QAsymmS8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt8_2, DepthToSpaceTest2<DataType::QAsymmS8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt8_3, DepthToSpaceTest3<DataType::QAsymmS8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt8_4, DepthToSpaceTest4<DataType::QAsymmS8>, DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcUint8_1, DepthToSpaceTest1<DataType::QAsymmU8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcUint8_2, DepthToSpaceTest2<DataType::QAsymmU8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcUint8_3, DepthToSpaceTest3<DataType::QAsymmU8>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcUint8_4, DepthToSpaceTest4<DataType::QAsymmU8>, DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt16_1, DepthToSpaceTest1<DataType::QSymmS16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt16_2, DepthToSpaceTest2<DataType::QSymmS16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt16_3, DepthToSpaceTest3<DataType::QSymmS16>, DataLayout::NHWC);
ARMNN_AUTO_TEST_CASE(DepthToSpaceNhwcInt16_4, DepthToSpaceTest4<DataType::QSymmS16>, DataLayout::NHWC);

// SpaceToDepth
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchwAsymmQ8, SpaceToDepthNchwAsymmQ8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwcAsymmQ8, SpaceToDepthNhwcAsymmQ8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwc1Float32, SpaceToDepthNhwcFloat32Test1)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchw1Float32, SpaceToDepthNchwFloat32Test1)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwc2Float32, SpaceToDepthNhwcFloat32Test2)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchw2Float32, SpaceToDepthNchwFloat32Test2)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwcQSymm16, SpaceToDepthNhwcQSymm16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchwQSymm16, SpaceToDepthNchwQSymm16Test)

// Strided Slice
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dFloat32, StridedSlice4dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dReverseFloat32, StridedSlice4dReverseFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleStrideFloat32, StridedSliceSimpleStrideFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleRangeMaskFloat32, StridedSliceSimpleRangeMaskFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskFloat32, StridedSliceShrinkAxisMaskFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskCTSFloat32, StridedSliceShrinkAxisMaskCTSFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskBitPosition0Dim3Float32,
                     StridedSliceShrinkAxisMaskBitPosition0Dim3Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0Float32, StridedSliceShrinkAxisMaskBitPosition0Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition1Float32, StridedSliceShrinkAxisMaskBitPosition1Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition2Float32, StridedSliceShrinkAxisMaskBitPosition2Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition3Float32, StridedSliceShrinkAxisMaskBitPosition3Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0And1Float32, StridedSliceShrinkAxisMaskBitPosition0And1Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0And2Float32, StridedSliceShrinkAxisMaskBitPosition0And2Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0And3Float32, StridedSliceShrinkAxisMaskBitPosition0And3Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0And1And3Float32, StridedSliceShrinkAxisMaskBitPosition0And1And3Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dFloat32, StridedSlice3dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dReverseFloat32, StridedSlice3dReverseFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dFloat32, StridedSlice2dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dReverseFloat32, StridedSlice2dReverseFloat32Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dUint8, StridedSlice4dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dReverseUint8, StridedSlice4dReverseUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleStrideUint8, StridedSliceSimpleStrideUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleRangeMaskUint8, StridedSliceSimpleRangeMaskUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskUint8, StridedSliceShrinkAxisMaskUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0Dim3Uint8, StridedSliceShrinkAxisMaskBitPosition0Dim3Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0Uint8, StridedSliceShrinkAxisMaskBitPosition0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition1Uint8, StridedSliceShrinkAxisMaskBitPosition1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition2Uint8, StridedSliceShrinkAxisMaskBitPosition2Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition3Uint8, StridedSliceShrinkAxisMaskBitPosition3Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0And1Uint8, StridedSliceShrinkAxisMaskBitPosition0And1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0And2Uint8, StridedSliceShrinkAxisMaskBitPosition0And2Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0And3Uint8, StridedSliceShrinkAxisMaskBitPosition0And3Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0And1And3Uint8, StridedSliceShrinkAxisMaskBitPosition0And1And3Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dUint8, StridedSlice3dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dReverseUint8, StridedSlice3dReverseUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dUint8, StridedSlice2dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dReverseUint8, StridedSlice2dReverseUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dInt16, StridedSlice4dInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dReverseInt16, StridedSlice4dReverseInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleStrideInt16, StridedSliceSimpleStrideInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleRangeMaskInt16, StridedSliceSimpleRangeMaskInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskInt16, StridedSliceShrinkAxisMaskInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dInt16, StridedSlice3dInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice3dReverseInt16, StridedSlice3dReverseInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dInt16, StridedSlice2dInt16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice2dReverseInt16, StridedSlice2dReverseInt16Test)

// Debug
ARMNN_AUTO_TEST_CASE(Debug4dFloat32, Debug4dFloat32Test)
ARMNN_AUTO_TEST_CASE(Debug3dFloat32, Debug3dFloat32Test)
ARMNN_AUTO_TEST_CASE(Debug2dFloat32, Debug2dFloat32Test)
ARMNN_AUTO_TEST_CASE(Debug1dFloat32, Debug1dFloat32Test)

ARMNN_AUTO_TEST_CASE(Debug4dBFloat16, Debug4dBFloat16Test)
ARMNN_AUTO_TEST_CASE(Debug3dBFloat16, Debug3dBFloat16Test)
ARMNN_AUTO_TEST_CASE(Debug2dBFloat16, Debug2dBFloat16Test)
ARMNN_AUTO_TEST_CASE(Debug1dBFloat16, Debug1dBFloat16Test)

ARMNN_AUTO_TEST_CASE(Debug4dUint8, Debug4dUint8Test)
ARMNN_AUTO_TEST_CASE(Debug3dUint8, Debug3dUint8Test)
ARMNN_AUTO_TEST_CASE(Debug2dUint8, Debug2dUint8Test)
ARMNN_AUTO_TEST_CASE(Debug1dUint8, Debug1dUint8Test)

ARMNN_AUTO_TEST_CASE(Debug4dQSymm16, Debug4dInt16Test)
ARMNN_AUTO_TEST_CASE(Debug3dQSymm16, Debug3dInt16Test)
ARMNN_AUTO_TEST_CASE(Debug2dQSymm16, Debug2dInt16Test)
ARMNN_AUTO_TEST_CASE(Debug1dQSymm16, Debug1dInt16Test)

// Gather
ARMNN_AUTO_TEST_CASE(Gather1dParamsFloat32, Gather1dParamsFloat32Test)
ARMNN_AUTO_TEST_CASE(Gather1dParamsFloat16, Gather1dParamsFloat16Test)
ARMNN_AUTO_TEST_CASE(Gather1dParamsUint8, Gather1dParamsUint8Test)
ARMNN_AUTO_TEST_CASE(Gather1dParamsInt16, Gather1dParamsInt16Test)
ARMNN_AUTO_TEST_CASE(Gather1dParamsInt32, Gather1dParamsInt32Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsFloat32, GatherMultiDimParamsFloat32Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsFloat16, GatherMultiDimParamsFloat16Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsUint8, GatherMultiDimParamsUint8Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsInt16, GatherMultiDimParamsInt16Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsInt32, GatherMultiDimParamsInt32Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsMultiDimIndicesFloat32, GatherMultiDimParamsMultiDimIndicesFloat32Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsMultiDimIndicesFloat16, GatherMultiDimParamsMultiDimIndicesFloat16Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsMultiDimIndicesUint8, GatherMultiDimParamsMultiDimIndicesUint8Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsMultiDimIndicesInt16, GatherMultiDimParamsMultiDimIndicesInt16Test)
ARMNN_AUTO_TEST_CASE(GatherMultiDimParamsMultiDimIndicesInt32, GatherMultiDimParamsMultiDimIndicesInt32Test)

// Abs
ARMNN_AUTO_TEST_CASE(Abs2d, Abs2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Abs3d, Abs3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(AbsZero, AbsZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Abs2dFloat16, Abs2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(Abs3dFloat16, Abs3dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(Abs2dSigned32, Abs2dTest<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE(Abs3dSigned32, Abs3dTest<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE(AbsZeroSigned32, AbsZeroTest<DataType::Signed32>)

ARMNN_AUTO_TEST_CASE(Abs2dQuantisedAsymmS8, Abs2dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(Abs3dQuantisedAsymmS8, Abs3dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(Abs2dQuantisedAsymm8, Abs2dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(Abs3dQuantisedAsymm8, Abs3dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(Abs2dQuantisedSymm16, Abs2dTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(Abs3dQuantisedSymm16, Abs3dTest<DataType::QSymmS16>)

// Detection PostProcess
BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsFloat)
{
    DetectionPostProcessRegularNmsFloatTest<RefWorkloadFactory>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsFloat)
{
    DetectionPostProcessFastNmsFloatTest<RefWorkloadFactory>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsInt8)
{
    DetectionPostProcessRegularNmsQuantizedTest<
        RefWorkloadFactory, DataType::QAsymmS8>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsInt8)
{
    DetectionPostProcessRegularNmsQuantizedTest<
        RefWorkloadFactory, DataType::QAsymmS8>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsUint8)
{
    DetectionPostProcessRegularNmsQuantizedTest<
        RefWorkloadFactory, DataType::QAsymmU8>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsUint8)
{
    DetectionPostProcessRegularNmsQuantizedTest<
        RefWorkloadFactory, DataType::QAsymmU8>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessRegularNmsInt16)
{
    DetectionPostProcessRegularNmsQuantizedTest<
        RefWorkloadFactory, DataType::QSymmS16>();
}
BOOST_AUTO_TEST_CASE(DetectionPostProcessFastNmsInt16)
{
    DetectionPostProcessFastNmsQuantizedTest<
        RefWorkloadFactory, DataType::QSymmS16>();
}

// Dequantize
ARMNN_AUTO_TEST_CASE(DequantizeSimpleUint8, DequantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE(DequantizeOffsetUint8, DequantizeOffsetUint8Test)
ARMNN_AUTO_TEST_CASE(DequantizeSimpleAsymmInt8, DequantizeSimpleAsymmInt8Test)
ARMNN_AUTO_TEST_CASE(DequantizeOffsetAsymmInt8, DequantizeOffsetAsymmInt8Test)
ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt8, DequantizeSimpleInt8Test)
ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt16, DequantizeSimpleInt16Test)
ARMNN_AUTO_TEST_CASE(DequantizeSimpleUint8ToFp16, DequantizeSimpleUint8ToFp16Test)
ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt8ToFp16, DequantizeSimpleInt8ToFp16Test)
ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt16ToFp16, DequantizeSimpleInt16ToFp16Test)

// Quantize
ARMNN_AUTO_TEST_CASE(QuantizeSimpleUint8, QuantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE(QuantizeClampUint8, QuantizeClampUint8Test)
ARMNN_AUTO_TEST_CASE(QuantizeClampAsymmInt8, QuantizeClampAsymmInt8Test)
ARMNN_AUTO_TEST_CASE(QuantizeClampInt8, QuantizeClampInt8Test)
ARMNN_AUTO_TEST_CASE(QuantizeClampInt16, QuantizeClampInt16Test)

// PReLU
ARMNN_AUTO_TEST_CASE(PreluFloat32, PreluTest<RefWorkloadFactory, DataType::Float32>)
ARMNN_AUTO_TEST_CASE(PreluFloat16, PreluTest<RefWorkloadFactory, DataType::Float16>)
ARMNN_AUTO_TEST_CASE(PreluUint8,   PreluTest<RefWorkloadFactory, DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(PreluInt16,   PreluTest<RefWorkloadFactory, DataType::QSymmS16>)

// Slice
ARMNN_AUTO_TEST_CASE(Slice4dFloat32, Slice4dFloat32Test)
ARMNN_AUTO_TEST_CASE(Slice3dFloat32, Slice3dFloat32Test)
ARMNN_AUTO_TEST_CASE(Slice2dFloat32, Slice2dFloat32Test)
ARMNN_AUTO_TEST_CASE(Slice1dFloat32, Slice1dFloat32Test)

ARMNN_AUTO_TEST_CASE(Slice4dUint8, Slice4dUint8Test)
ARMNN_AUTO_TEST_CASE(Slice3dUint8, Slice3dUint8Test)
ARMNN_AUTO_TEST_CASE(Slice2dUint8, Slice2dUint8Test)
ARMNN_AUTO_TEST_CASE(Slice1dUint8, Slice1dUint8Test)

ARMNN_AUTO_TEST_CASE(Slice4dInt16, Slice4dInt16Test)
ARMNN_AUTO_TEST_CASE(Slice3dInt16, Slice3dInt16Test)
ARMNN_AUTO_TEST_CASE(Slice2dInt16, Slice2dInt16Test)
ARMNN_AUTO_TEST_CASE(Slice1dInt16, Slice1dInt16Test)

// Transpose
ARMNN_AUTO_TEST_CASE(SimpleTransposeBFloat16, SimpleTransposeTest<RefWorkloadFactory, DataType::BFloat16>)
ARMNN_AUTO_TEST_CASE(TransposeBFloat16ValueSet1Test, TransposeValueSet1Test<RefWorkloadFactory, DataType::BFloat16>)
ARMNN_AUTO_TEST_CASE(TransposeBFloat16ValueSet2Test, TransposeValueSet2Test<RefWorkloadFactory, DataType::BFloat16>)
ARMNN_AUTO_TEST_CASE(TransposeBFloat16ValueSet3Test, TransposeValueSet3Test<RefWorkloadFactory, DataType::BFloat16>)
ARMNN_AUTO_TEST_CASE(SimpleTransposeFloat32, SimpleTransposeTest<RefWorkloadFactory, DataType::Float32>)
ARMNN_AUTO_TEST_CASE(TransposeFloat32ValueSet1Test, TransposeValueSet1Test<RefWorkloadFactory, DataType::Float32>)
ARMNN_AUTO_TEST_CASE(TransposeFloat32ValueSet2Test, TransposeValueSet2Test<RefWorkloadFactory, DataType::Float32>)
ARMNN_AUTO_TEST_CASE(TransposeFloat32ValueSet3Test, TransposeValueSet3Test<RefWorkloadFactory, DataType::Float32>)
ARMNN_AUTO_TEST_CASE(SimpleTransposeQASymmS8, SimpleTransposeTest<RefWorkloadFactory, DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(TransposeQASymmS8ValueSet1Test, TransposeValueSet1Test<RefWorkloadFactory, DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(TransposeQASymmS8ValueSet2Test, TransposeValueSet2Test<RefWorkloadFactory, DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(TransposeQASymmS8ValueSet3Test, TransposeValueSet3Test<RefWorkloadFactory, DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(SimpleTransposeQASymm8, SimpleTransposeTest<RefWorkloadFactory, DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(TransposeQASymm8ValueSet1Test, TransposeValueSet1Test<RefWorkloadFactory, DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(TransposeQASymm8ValueSet2Test, TransposeValueSet2Test<RefWorkloadFactory, DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(TransposeQASymm8ValueSet3Test, TransposeValueSet3Test<RefWorkloadFactory, DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(SimpleTransposeQSymm16, SimpleTransposeTest<RefWorkloadFactory, DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(TransposeQSymm16ValueSet1Test, TransposeValueSet1Test<RefWorkloadFactory, DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(TransposeQSymm16ValueSet2Test, TransposeValueSet2Test<RefWorkloadFactory, DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(TransposeQSymm16ValueSet3Test, TransposeValueSet3Test<RefWorkloadFactory, DataType::QSymmS16>)

// TransposeConvolution2d
ARMNN_AUTO_TEST_CASE(SimpleTransposeConvolution2dFloatNchw,
                     SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleTransposeConvolution2dFloatNhwc,
                     SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleTransposeConvolution2dInt8Nchw,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleTransposeConvolution2dInt8Nhwc,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleTransposeConvolution2dUint8Nchw,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleTransposeConvolution2dUint8Nhwc,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(SimpleTransposeConvolution2dInt16Nchw,
                     SimpleTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(SimpleTransposeConvolution2dInt16Nhwc,
                     SimpleTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(UnbiasedSimpleTransposeConvolution2dFloatNchw,
                     SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedSimpleTransposeConvolution2dFloatNhwc,
                     SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedSimpleTransposeConvolution2dInt8Nchw,
                    SimpleTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedSimpleTransposeConvolution2dInt8Nhwc,
                    SimpleTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedSimpleTransposeConvolution2dUint8Nchw,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedSimpleTransposeConvolution2dUint8Nhwc,
                     SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedSimpleTransposeConvolution2dInt16Nchw,
                     SimpleTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedSimpleTransposeConvolution2dInt16Nhwc,
                     SimpleTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(PaddedTransposeConvolution2dFloatNchw,
                     PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(PaddedTransposeConvolution2dFloatNhwc,
                     PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(PaddedTransposeConvolution2dInt8Nchw,
                    PaddedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(PaddedTransposeConvolution2dInt8Nhwc,
                    PaddedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(PaddedTransposeConvolution2dUint8Nchw,
                     PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(PaddedTransposeConvolution2dUint8Nhwc,
                     PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(PaddedTransposeConvolution2dInt16Nchw,
                     PaddedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(PaddedTransposeConvolution2dInt16Nhwc,
                     PaddedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(UnbiasedPaddedTransposeConvolution2dFloatNchw,
                     PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedPaddedTransposeConvolution2dFloatNhwc,
                     PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedPaddedTransposeConvolution2dInt8Nchw,
                    PaddedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedPaddedTransposeConvolution2dInt8Nhwc,
                    PaddedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedPaddedTransposeConvolution2dUint8Nchw,
                     PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedPaddedTransposeConvolution2dUint8Nhwc,
                     PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedPaddedTransposeConvolution2dInt16Nchw,
                     PaddedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedPaddedTransposeConvolution2dInt16Nhwc,
                     PaddedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(StridedTransposeConvolution2dFloatNchw,
                     StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(StridedTransposeConvolution2dFloatNhwc,
                     StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(StridedTransposeConvolution2dInt8Nchw,
                    StridedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(StridedTransposeConvolution2dInt8Nhwc,
                    StridedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(StridedTransposeConvolution2dUint8Nchw,
                     StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(StridedTransposeConvolution2dUint8Nhwc,
                     StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(StridedTransposeConvolution2dInt16Nchw,
                     StridedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(StridedTransposeConvolution2dInt16Nhwc,
                     StridedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(UnbiasedStridedTransposeConvolution2dFloatNchw,
                     StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedStridedTransposeConvolution2dFloatNhwc,
                     StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedStridedTransposeConvolution2dInt8Nchw,
                    StridedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedStridedTransposeConvolution2dInt8Nhwc,
                    StridedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    true,
                    DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedStridedTransposeConvolution2dUint8Nchw,
                     StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedStridedTransposeConvolution2dUint8Nhwc,
                     StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     true,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(UnbiasedStridedTransposeConvolution2dInt16Nchw,
                     StridedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(UnbiasedStridedTransposeConvolution2dInt16Nhwc,
                     StridedTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     true,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(MultiChannelTransposeConvolution2dFloatNchw,
                     MultiChannelTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(MultiChannelTransposeConvolution2dFloatNhwc,
                     MultiChannelTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(MultiChannelTransposeConvolution2dInt8Nchw,
                    MultiChannelTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(MultiChannelTransposeConvolution2dInt8Nhwc,
                    MultiChannelTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                    DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(MultiChannelTransposeConvolution2dUint8Nchw,
                     MultiChannelTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(MultiChannelTransposeConvolution2dUint8Nhwc,
                     MultiChannelTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE(MultiChannelTransposeConvolution2dInt16Nchw,
                     MultiChannelTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE(MultiChannelTransposeConvolution2dInt16Nhwc,
                     MultiChannelTransposeConvolution2dTest<DataType::QSymmS16, DataType::Signed32>,
                     DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE(TransposeConvolution2dPerAxisQuantTestNchw,
                     TransposeConvolution2dPerAxisQuantTest,
                     DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE(TransposeConvolution2dPerAxisQuantTestNhwc,
                     TransposeConvolution2dPerAxisQuantTest,
                     DataLayout::NHWC);

// Stack
ARMNN_AUTO_TEST_CASE_WITH_THF(Stack0Axis,           StackAxis0Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput4DAxis1,   StackOutput4DAxis1Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput4DAxis2,   StackOutput4DAxis2Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput4DAxis3,   StackOutput4DAxis3Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput3DInputs3, StackOutput3DInputs3Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput5D,        StackOutput5DFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackFloat16,         StackFloat16Test)

// Neg
ARMNN_AUTO_TEST_CASE(Neg2d, Neg2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Neg3d, Neg3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(NegZero, NegZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(NegNegative, NegNegativeTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Neg2dFloat16, Neg2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(Neg3dFloat16, Neg3dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(Neg2dQuantisedAsymmS8, Neg2dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(Neg3dQuantisedAsymmS8, Neg3dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(Neg2dQuantisedAsymm8, Neg2dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(Neg3dQuantisedAsymm8, Neg3dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(Neg2dQuantisedSymm16, Neg2dTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(Neg3dQuantisedSymm16, Neg3dTest<DataType::QSymmS16>)

// Exp
ARMNN_AUTO_TEST_CASE(Exp2d, Exp2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Exo3d, Exp3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(ExpZero, ExpZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(ExpNegative, ExpNegativeTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE(Exp2dFloat16, Exp2dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(Exp3dFloat16, Exp3dTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE(Exp2dQuantisedAsymmS8, Exp2dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(Exp3dQuantisedAsymmS8, Exp3dTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE(Exp2dQuantisedAsymm8, Exp2dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(Exp3dQuantisedAsymm8, Exp3dTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE(Exp2dQuantisedSymm16, Exp2dTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE(Exp3dQuantisedSymm16, Exp3dTest<DataType::QSymmS16>)

BOOST_AUTO_TEST_SUITE_END()
