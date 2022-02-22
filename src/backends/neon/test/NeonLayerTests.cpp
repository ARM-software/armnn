//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "NeonWorkloadFactoryHelper.hpp"

#include <armnnTestUtils/TensorHelpers.hpp>
#include <UnitTests.hpp>

#include <neon/NeonLayerSupport.hpp>
#include <neon/NeonWorkloadFactory.hpp>

#include <backendsCommon/test/ActivationFixture.hpp>
#include <backendsCommon/test/LayerTests.hpp>

#include <doctest/doctest.h>

TEST_SUITE("Compute_ArmComputeNeon")
{
using namespace armnn;

using FactoryType = NeonWorkloadFactory;

// ============================================================================
// UNIT tests

// BatchToSpace
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat321, BatchToSpaceNdNhwcTest1<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat322, BatchToSpaceNdNhwcTest2<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcFloat323, BatchToSpaceNdNhwcTest3<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat321, BatchToSpaceNdNchwTest1<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat322, BatchToSpaceNdNchwTest2<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwFloat323, BatchToSpaceNdNchwTest3<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcInt1, BatchToSpaceNdNhwcTest1<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcInt2, BatchToSpaceNdNhwcTest2<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcInt3, BatchToSpaceNdNhwcTest3<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwInt1, BatchToSpaceNdNchwTest1<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwInt2, BatchToSpaceNdNchwTest2<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwInt3, BatchToSpaceNdNchwTest3<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint1, BatchToSpaceNdNhwcTest1<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint2, BatchToSpaceNdNhwcTest2<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNhwcUint3, BatchToSpaceNdNhwcTest3<DataType::QAsymmU8>)

ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint1, BatchToSpaceNdNchwTest1<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint2, BatchToSpaceNdNchwTest2<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchToSpaceNdNchwUint3, BatchToSpaceNdNchwTest3<DataType::QAsymmU8>)

// Convolution
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution1d, Convolution1dTest, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d, SimpleConvolution2d3x5Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dNhwc, SimpleConvolution2d3x5Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3Uint8, SimpleConvolution2d3x3Uint8Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2d3x3Uint8Nhwc, SimpleConvolution2d3x3Uint8Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2d, SimpleConvolution2d3x5Test, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dNhwc, SimpleConvolution2d3x5Test, false, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dStride2x2Nhwc,
                     SimpleConvolution2d3x3Stride2x2Test, false, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dSquare, SimpleConvolution2d3x3Test, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPadding,
                              Convolution2dAsymmetricPaddingTest,
                              DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedConvolution2dSquareNhwc, SimpleConvolution2d3x3Test, false, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dAsymmetricPaddingNhwc,
                     Convolution2dAsymmetricPaddingTest,
                     DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConvolution2dSquareNhwc, SimpleConvolution2d3x3NhwcTest, false)

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
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test
                             <DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Nhwc,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test
                             <DataType::Float32, DataType::Float32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Int8,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test
                             <DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3NhwcInt8,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test
                             <DataType::QAsymmS8, DataType::Signed32>,
                     false,
                     DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3Uint8,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test
                             <DataType::QAsymmU8, DataType::Signed32>,
                     false,
                     DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2d2x2Dilation2x2Padding2x2Stride3x3NhwcUint8,
                     Convolution2d2x2Dilation2x2Padding2x2Stride3x3Test
                             <DataType::QAsymmU8, DataType::Signed32>,
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

ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2dPerAxisQuantTestNchw, Convolution2dPerAxisQuantTest, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution2dPerAxisQuantTestNhwc, Convolution2dPerAxisQuantTest, DataLayout::NHWC);

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
#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3d2x3x3TestFloat16,
                              Convolution3d2x3x3Float16Test,
                              true,
                              DataLayout::NDHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(Convolution3d2x2x2SmallTestFloat16,
                              Convolution3d2x2x2SmallFloat16Test,
                              false,
                              DataLayout::NDHWC)
#endif

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

// Depthwise Convolution
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2d, DepthwiseConvolution2dTest, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dUint8, DepthwiseConvolution2dUint8Test, true, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2d, DepthwiseConvolution2dTest, false, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1,
                     DepthwiseConvolution2dDepthMul1Test, false, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test, true, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Uint8,
                     DepthwiseConvolution2dDepthMul1Uint8Test, false, DataLayout::NCHW)

// NHWC Depthwise Convolution
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1NHhwc,
                     DepthwiseConvolution2dDepthMul1Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Nhwc,
                     DepthwiseConvolution2dDepthMul1Test, false, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test, true, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedDepthwiseConvolution2dDepthMul1Uint8Nhwc,
                     DepthwiseConvolution2dDepthMul1Uint8Test, false, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(DepthwiseConvolution2dDepthNhwc, DepthwiseConvolution2dDepthNhwcTest, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleDepthwiseConvolution2d3x3Dilation3x3Nhwc,
                     SimpleDepthwiseConvolution2d3x3Dilation3x3NhwcTest)


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

namespace
{

DepthwiseConvolution2dDescriptor MakeDepthwiseConv2dDesc(uint32_t strideX, uint32_t strideY,
    uint32_t depthMultiplier = 1, uint32_t padLeft = 0, uint32_t padRight = 0,
    uint32_t padTop = 0, uint32_t padBottom = 0)
{
    IgnoreUnused(depthMultiplier);

    DepthwiseConvolution2dDescriptor desc;

    desc.m_PadLeft = padLeft;
    desc.m_PadRight = padRight;

    desc.m_PadTop = padTop;
    desc.m_PadBottom = padBottom;
    desc.m_StrideX = strideX;
    desc.m_StrideY = strideY;
    desc.m_BiasEnabled = false;

    return desc;
}

TensorInfo CreateOutputTensorInfo(const TensorInfo& inputInfo,
                                         const TensorInfo& weightsInfo,
                                         const DepthwiseConvolution2dDescriptor& descriptor,
                                         DataType dataType)
{
    const TensorShape& inputShape  = inputInfo.GetShape();
    const TensorShape& filterShape = weightsInfo.GetShape();

    unsigned int inWidth = inputShape[3];
    unsigned int inHeight = inputShape[2];
    unsigned int inBatchSize = inputShape[0];

    unsigned int filterWidth = filterShape[2];
    unsigned int readWidth = (inWidth + descriptor.m_PadLeft + descriptor.m_PadRight) - (filterWidth);
    unsigned int outWidth =  1u + (readWidth / descriptor.m_StrideX);

    unsigned int filterHeight = filterShape[1];
    unsigned int readHeight = (inHeight + descriptor.m_PadTop + descriptor.m_PadBottom) - (filterHeight);
    unsigned int outHeight = 1u + (readHeight / descriptor.m_StrideY);

    unsigned int outChannels = filterShape[3];
    unsigned int outBatchSize = inBatchSize;

    TensorShape outputShape({outBatchSize, outChannels, outHeight, outWidth});
    return TensorInfo(outputShape, dataType);
}
}

TEST_CASE("DepthwiseConv2dUtils")
{
    const DataType dataType = DataType::Float32;

    TensorInfo inputInfo({1, 1, 10, 10 }, dataType);
    TensorInfo outputInfo;
    TensorInfo weightsInfo3x3({ 1, 3, 3, 1 }, dataType); // [1,H,W,I*M]
    TensorInfo biasesInfo;

    DepthwiseConvolution2dDescriptor descriptor;
    NeonLayerSupport layerSupport;

    // Strides supported: 1,2,3
    descriptor = MakeDepthwiseConv2dDesc(1, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(1, 2);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(1, 3);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(2, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(2, 2);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(2, 3);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(3, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(3, 2);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    descriptor = MakeDepthwiseConv2dDesc(3, 3);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    // Supported stride 4
    descriptor = MakeDepthwiseConv2dDesc(4, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));

    // Supported weights shape 1x1
    TensorInfo weightsInfo1x1({ 1, 1, 1, 1 }, DataType::Float32);
    descriptor = MakeDepthwiseConv2dDesc(1, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo1x1, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo1x1, biasesInfo));

    // Supported shape 2x2
    TensorInfo weightsInfo2x2({ 1, 2, 2, 1 }, DataType::Float32);
    descriptor = MakeDepthwiseConv2dDesc(1, 1);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo2x2, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo2x2, biasesInfo));

    // Asymmetric padding
    descriptor = MakeDepthwiseConv2dDesc(1, 1, 1, 1, 2, 1, 2);
    outputInfo = CreateOutputTensorInfo(inputInfo, weightsInfo3x3, descriptor, dataType);
    CHECK(layerSupport.IsDepthwiseConvolutionSupported(inputInfo, outputInfo, descriptor,
                                                            weightsInfo3x3, biasesInfo));
}

// Dequantize
// Fp16 is only supported if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC is enabled
ARMNN_AUTO_TEST_CASE(DequantizeSimpleUint8, DequantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE(DequantizeOffsetUint8, DequantizeOffsetUint8Test)
ARMNN_AUTO_TEST_CASE(DequantizeSimpleInt16, DequantizeSimpleInt16Test)

// Pooling
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dSize3x3Stride2x4,
                                          SimpleMaxPooling2dSize3x3Stride2x4Test, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dSize3x3Stride2x4Uint8,
                                          SimpleMaxPooling2dSize3x3Stride2x4Uint8Test, true)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2d, SimpleMaxPooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dNhwc, SimpleMaxPooling2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dUint8, SimpleMaxPooling2dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaxPooling2dUint8Nhwc, SimpleMaxPooling2dUint8Test, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2d, SimpleAveragePooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2dNhwc, SimpleAveragePooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2dUint8,
                                          SimpleAveragePooling2dUint8Test, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAveragePooling2dUint8Nhwc,
                                          SimpleAveragePooling2dUint8Test, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(LargeTensorsAveragePooling2d, LargeTensorsAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LargeTensorsAveragePooling2dUint8, LargeTensorsAveragePooling2dUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleL2Pooling2d, SimpleL2Pooling2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleL2Pooling2dNeon, SimpleL2Pooling2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UNSUPPORTED_SimpleL2Pooling2dUint8, SimpleL2Pooling2dUint8Test, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(L2Pooling2dSize3Stride1, L2Pooling2dSize3Stride1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(UNSUPPORTED_L2Pooling2dSize3Stride1Uint8, L2Pooling2dSize3Stride1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Pooling2dSize3Stride3, L2Pooling2dSize3Stride3Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(UNSUPPORTED_L2Pooling2dSize3Stride3Uint8, L2Pooling2dSize3Stride3Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Pooling2dSize3Stride4, L2Pooling2dSize3Stride4Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(UNSUPPORTED_L2Pooling2dSize3Stride4Uint8, L2Pooling2dSize3Stride4Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Pooling2dSize7, L2Pooling2dSize7Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(UNSUPPORTED_L2Pooling2dSize7Uint8, L2Pooling2dSize7Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Pooling2dSize9, L2Pooling2dSize9Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(UNSUPPORTED_L2Pooling2dSize9Uint8, L2Pooling2dSize9Uint8Test)

// Ignore padding values for pooling but count padding fields into the divisor
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleMaxPooling2d, IgnorePaddingSimpleMaxPooling2dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleMaxPooling2dUint8, IgnorePaddingSimpleMaxPooling2dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingMaxPooling2dSize3, IgnorePaddingMaxPooling2dSize3Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingMaxPooling2dSize3Uint8, IgnorePaddingMaxPooling2dSize3Uint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2d, IgnorePaddingSimpleAveragePooling2dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2dUint8, IgnorePaddingSimpleAveragePooling2dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2dNoPadding,
                              IgnorePaddingSimpleAveragePooling2dNoPaddingTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleAveragePooling2dNoPaddingUint8,
                              IgnorePaddingSimpleAveragePooling2dNoPaddingUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingAveragePooling2dSize3, IgnorePaddingAveragePooling2dSize3Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingAveragePooling2dSize3Uint8,
                              IgnorePaddingAveragePooling2dSize3Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingAveragePooling2dSize3x2Stride2x2,
                              IgnorePaddingAveragePooling2dSize3x2Stride2x2Test, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingAveragePooling2dSize3x2Stride2x2NoPadding,
                              IgnorePaddingAveragePooling2dSize3x2Stride2x2Test,
                              true)

ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingSimpleL2Pooling2d, IgnorePaddingSimpleL2Pooling2dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(UNSUPPORTED_IgnorePaddingSimpleL2Pooling2dUint8,
                                          IgnorePaddingSimpleL2Pooling2dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(IgnorePaddingL2Pooling2dSize3, IgnorePaddingL2Pooling2dSize3Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(UNSUPPORTED_IgnorePaddingL2Pooling2dSize3Uint8,
                                          IgnorePaddingL2Pooling2dSize3Uint8Test)

// Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(ConstantLinearActivation, ConstantLinearActivationTest)

// Sigmoid Activation / Logistic
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSigmoid, SimpleSigmoidTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSigmoidUint8, SimpleSigmoidUint8Test)

// BoundedReLU Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu1, BoundedReLuUpperAndLowerBoundTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu6, BoundedReLuUpperBoundOnlyTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu1Uint8, BoundedReLuUint8UpperAndLowerBoundTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu6Uint8, BoundedReLuUint8UpperBoundOnlyTest)

// ReLU Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLu, ReLuTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReLuUint8, ReLuUint8Test)

// SoftReLU Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(SoftReLu, SoftReLuTest)

// LeakyReLU Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(LeakyReLu, LeakyReLuTest)

// Abs Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs, AbsTest)

// Sqrt Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(Sqrt, SqrtTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SqrtNN, SqrtNNTest)

// Square Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(Square, SquareTest)

// Tanh Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(Tanh, TanhTest)

// Elu Activation
ARMNN_AUTO_TEST_CASE_WITH_THF(Elu, EluTest)

// Softmax
// Moved to NeonLayerTests_NDK_Bug.cpp
//ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta1, SimpleSoftmaxTest, 1.0f)
//ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta2, SimpleSoftmaxTest, 2.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta1Uint8, SimpleSoftmaxUint8Test, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSoftmaxBeta2Uint8, SimpleSoftmaxUint8Test, 2.0f)

// LogSoftmax
// Moved to NeonLayerTests_NDK_Bug.cpp
//ARMNN_AUTO_TEST_CASE_WITH_THF(LogSoftmaxFloat32_1, LogSoftmaxTest1<DataType::Float32>)

// Space To Batch Nd
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleFloat32, SpaceToBatchNdSimpleFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsFloat32, SpaceToBatchNdMultiChannelsFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockFloat32, SpaceToBatchNdMultiBlockFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingFloat32, SpaceToBatchNdPaddingFloat32Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleUint8, SpaceToBatchNdSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsUint8, SpaceToBatchNdMultiChannelsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockUint8, SpaceToBatchNdMultiBlockUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingUint8, SpaceToBatchNdPaddingUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcFloat32, SpaceToBatchNdSimpleNhwcFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcFloat32,
                                          SpaceToBatchNdMultiChannelsNhwcFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcFloat32, SpaceToBatchNdMultiBlockNhwcFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcFloat32, SpaceToBatchNdPaddingNhwcFloat32Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdSimpleNhwcUint8, SpaceToBatchNdSimpleNhwcUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiChannelsNhwcUint8,
                                          SpaceToBatchNdMultiChannelsNhwcUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdMultiBlockNhwcUint8, SpaceToBatchNdMultiBlockNhwcUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToBatchNdPaddingNhwcUint8, SpaceToBatchNdPaddingNhwcUint8Test)

// SpaceToDepth
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchwAsymmQ8, SpaceToDepthNchwAsymmQ8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwcsymmQ8, SpaceToDepthNhwcAsymmQ8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwc1Float32, SpaceToDepthNhwcFloat32Test1)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchw1Float32, SpaceToDepthNchwFloat32Test1)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwc2Float32, SpaceToDepthNhwcFloat32Test2)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchw2Float32, SpaceToDepthNchwFloat32Test2)

ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNhwcQSymm16, SpaceToDepthNhwcQSymm16Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SpaceToDepthNchwQSymm16, SpaceToDepthNchwQSymm16Test)

// Splitter
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterFloat32, SplitterFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSplitterUint8, SplitterUint8Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterFloat32, CopyViaSplitterFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(CopyViaSplitterUint8, CopyViaSplitterUint8Test)

// Concat
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleConcat, ConcatTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint8, ConcatUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConcatUint8DifferentInputOutputQParam,
                     ConcatDifferentInputOutputQParamTest<DataType::QAsymmU8>, false)

// Convert from BFloat16 to Float32
ARMNN_AUTO_TEST_CASE_WITH_THF(ConvertBf16ToFp32, ConvertBf16ToFp32Test)

// Convert from Float32 to BFloat16
ARMNN_AUTO_TEST_CASE_WITH_THF(ConvertFp32ToBf16, ConvertFp32ToBf16Test)

// Fully Connected
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFullyConnected, FullyConnectedFloat32Test, false, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFullyConnectedWithBias, FullyConnectedFloat32Test, true, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFullyConnectedWithTranspose, FullyConnectedFloat32Test, false, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedLarge, FullyConnectedLargeTest, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedLargeTransposed, FullyConnectedLargeTest, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedUint8, FullyConnectedTest<DataType::QAsymmU8>, false, true)
ARMNN_AUTO_TEST_CASE_WITH_THF(FullyConnectedBiasedUint8, FullyConnectedTest<DataType::QAsymmU8>, true, true)

// Add
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleAdd, AdditionTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Add5d, Addition5dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(AddBroadcast, AdditionBroadcastTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(AddBroadcast1Element, AdditionBroadcast1ElementTest)

// Sub
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleSub, SubtractionTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcast1Element, SubtractionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcast, SubtractionBroadcastTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SubtractionUint8, SubtractionUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcastUint8, SubtractionBroadcastUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(SubBroadcast1ElementUint8, SubtractionBroadcast1ElementUint8Test)

// Div
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleDivision, DivisionTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionByZero, DivisionByZeroTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionBroadcast1Element, DivisionBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(DivisionBroadcast1DVector, DivisionBroadcast1DVectorTest)

// Mul
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMultiplication, MultiplicationTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1Element, MultiplicationBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1DVector, MultiplicationBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationUint8, MultiplicationUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1ElementUint8, MultiplicationBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiplicationBroadcast1DVectorUint8, MultiplicationBroadcast1DVectorUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Multiplication5d, Multiplication5dTest)

// Batch Norm
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormFloat32, BatchNormFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(BatchNormFloat32Nhwc, BatchNormFloat32NhwcTest)

// Rank
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize1Float16,  RankDimSize1Test<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize1Float32,  RankDimSize1Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize1QAsymmU8, RankDimSize1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize1Signed32, RankDimSize1Test<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize1QSymmS16, RankDimSize1Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize1QAsymmS8, RankDimSize1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize1BFloat16, RankDimSize1Test<DataType::BFloat16>)

ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize2Float16,  RankDimSize2Test<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize2Float32,  RankDimSize2Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize2QAsymmU8, RankDimSize2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize2Signed32, RankDimSize2Test<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize2QSymmS16, RankDimSize2Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize2QAsymmS8, RankDimSize2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize2BFloat16, RankDimSize2Test<DataType::BFloat16>)

ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize3Float16,  RankDimSize3Test<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize3Float32,  RankDimSize3Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize3QAsymmU8, RankDimSize3Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize3Signed32, RankDimSize3Test<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize3QSymmS16, RankDimSize3Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize3QAsymmS8, RankDimSize3Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize3BFloat16, RankDimSize3Test<DataType::BFloat16>)

ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize4Float16,  RankDimSize4Test<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize4Float32,  RankDimSize4Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize4QAsymmU8, RankDimSize4Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize4Signed32, RankDimSize4Test<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize4QSymmS16, RankDimSize4Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize4QAsymmS8, RankDimSize4Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RankDimSize4BFloat16, RankDimSize4Test<DataType::BFloat16>)

// InstanceNormalization
ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat32Nchw, InstanceNormFloat32Test, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat32Nhwc, InstanceNormFloat32Test, DataLayout::NHWC);

ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat32Nchw2, InstanceNormFloat32Test2, DataLayout::NCHW);
ARMNN_AUTO_TEST_CASE_WITH_THF(InstanceNormFloat32Nhwc2, InstanceNormFloat32Test2, DataLayout::NHWC);

// Constant
ARMNN_AUTO_TEST_CASE_WITH_THF(Constant, ConstantTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(ConstantUint8, ConstantUint8SimpleQuantizationScaleNoOffsetTest)

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
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2, Concat3dDim2Test, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2Uint8, Concat3dDim2Uint8Test, false)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0DiffInputDims, Concat3dDim0DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim0DiffInputDimsUint8, Concat3dDim0DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1DiffInputDims, Concat3dDim1DiffInputDimsTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim1DiffInputDimsUint8, Concat3dDim1DiffInputDimsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2DiffInputDims, Concat3dDim2DiffInputDimsTest, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat3dDim2DiffInputDimsUint8, Concat3dDim2DiffInputDimsUint8Test, false)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim0, Concat4dDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim1, Concat4dDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim3, Concat4dDim3Test, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim0Uint8, Concat4dDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim1Uint8, Concat4dDim1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDim3Uint8, Concat4dDim3Uint8Test, false)

ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim0, Concat4dDiffShapeDim0Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim1, Concat4dDiffShapeDim1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim3, Concat4dDiffShapeDim3Test, false)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim0Uint8, Concat4dDiffShapeDim0Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim1Uint8, Concat4dDiffShapeDim1Uint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Concat4dDiffShapeDim3Uint8, Concat4dDiffShapeDim3Uint8Test, false)

// L2 Normalization
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1d, L2Normalization1dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2d, L2Normalization2dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization3d, L2Normalization3dTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization4d, L2Normalization4dTest, DataLayout::NCHW)

// Moved to NeonLayerTests_NDK_Bug.cpp
//ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization1dNhwc, L2Normalization1dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2dNhwc, L2Normalization2dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization3dNhwc, L2Normalization3dTest, DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization4dNhwc, L2Normalization4dTest, DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(L2Normalization2dShape, L2Normalization2dShapeTest);

ARMNN_AUTO_TEST_CASE_WITH_THF(L2NormalizationDefaultEpsilon, L2NormalizationDefaultEpsilonTest, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(L2NormalizationNonDefaultEpsilon, L2NormalizationNonDefaultEpsilonTest, DataLayout::NCHW)

// Floor
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFloor, SimpleFloorTest<DataType::Float32>)

// Gather
ARMNN_AUTO_TEST_CASE_WITH_THF(Gather1dParamsFloat32, Gather1dParamsFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(Gather1dParamsUint8, Gather1dParamsUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsFloat32, GatherMultiDimParamsFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GatherMultiDimParamsUint8, GatherMultiDimParamsUint8Test)

// Equal
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualSimple,            EqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1Element, EqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1dVector, EqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(EqualSimpleUint8,            EqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1ElementUint8, EqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(EqualBroadcast1dVectorUint8, EqualBroadcast1dVectorUint8Test)

// Greater
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterSimple,            GreaterSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1Element, GreaterBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1dVector, GreaterBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterSimpleUint8,            GreaterSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1ElementUint8, GreaterBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterBroadcast1dVectorUint8, GreaterBroadcast1dVectorUint8Test)

// GreaterOrEqual
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualSimple,            GreaterOrEqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1Element, GreaterOrEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1dVector, GreaterOrEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualSimpleUint8,            GreaterOrEqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1ElementUint8, GreaterOrEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(GreaterOrEqualBroadcast1dVectorUint8, GreaterOrEqualBroadcast1dVectorUint8Test)

// Less
ARMNN_AUTO_TEST_CASE_WITH_THF(LessSimple,            LessSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1Element, LessBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1dVector, LessBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessSimpleUint8,            LessSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1ElementUint8, LessBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessBroadcast1dVectorUint8, LessBroadcast1dVectorUint8Test)

// LessOrEqual
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualSimple,            LessOrEqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1Element, LessOrEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1dVector, LessOrEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualSimpleUint8,            LessOrEqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1ElementUint8, LessOrEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LessOrEqualBroadcast1dVectorUint8, LessOrEqualBroadcast1dVectorUint8Test)

// NotEqual
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualSimple,            NotEqualSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1Element, NotEqualBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1dVector, NotEqualBroadcast1dVectorTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualSimpleUint8,            NotEqualSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1ElementUint8, NotEqualBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(NotEqualBroadcast1dVectorUint8, NotEqualBroadcast1dVectorUint8Test)

// Reshape
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleReshapeFloat32, SimpleReshapeTest<armnn::DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleReshapeInt8, SimpleReshapeTest<armnn::DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleReshapeUint8, SimpleReshapeTest<armnn::DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Reshape5d, Reshape5dTest<armnn::DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReshapeBoolean, ReshapeBooleanTest)

// Pad - Constant
ARMNN_AUTO_TEST_CASE_WITH_THF(PadFloat322d, PadFloat322dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadFloat322dCustomPadding, PadFloat322dCustomPaddingTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadFloat323d, PadFloat323dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadFloat324d, PadFloat324dTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(PadUint82d, PadUint82dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadUint82dCustomPadding, PadUint82dCustomPaddingTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadUint83d, PadUint83dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadUint84d, PadUint84dTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(Pad2dQSymm16, Pad2dTestCommon<DataType::QSymmS16>, 2.0f, 0, 0.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Pad2dQSymm16CustomPadding, Pad2dTestCommon<DataType::QSymmS16>, 2.0f, 0, 1.0f)
ARMNN_AUTO_TEST_CASE_WITH_THF(Pad3dQSymm16, Pad3dTestCommon<DataType::QSymmS16>, 2.0f, 0)
ARMNN_AUTO_TEST_CASE_WITH_THF(Pad4dQSymm16, Pad4dTestCommon<DataType::QSymmS16>, 2.0f, 0)

// Pad - Symmetric & Reflect
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric2dFloat32, PadSymmetric2dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect2dFloat32, PadReflect2dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric3dFloat32, PadSymmetric3dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect3dFloat32, PadReflect3dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric3dUint8, PadSymmetric3dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect3dUint8, PadReflect3dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric3dInt8, PadSymmetric3dInt8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect3dInt8, PadReflect3dInt8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric4dFloat32, PadSymmetric4dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect4dFloat32, PadReflect4dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric4dUint8, PadSymmetric4dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect4dUint8, PadReflect4dUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadSymmetric4dInt8, PadSymmetric4dInt8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(PadReflect4dInt8, PadReflect4dInt8Test)

// Permute
ARMNN_AUTO_TEST_CASE_WITH_THF(SimplePermuteFloat32, SimplePermuteTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteFloat32ValueSet1Test, PermuteValueSet1Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteFloat32ValueSet2Test, PermuteValueSet2Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteFloat32ValueSet3Test, PermuteValueSet3Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimplePermuteQASymmS8, SimplePermuteTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQASymmS8ValueSet1Test, PermuteValueSet1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQASymmS8ValueSet2Test, PermuteValueSet2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQASymmS8ValueSet3Test, PermuteValueSet3Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimplePermuteQASymm8, SimplePermuteTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQASymm8ValueSet1Test, PermuteValueSet1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQASymm8ValueSet2Test, PermuteValueSet2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PermuteQASymm8ValueSet3Test, PermuteValueSet3Test<DataType::QAsymmU8>)

// Lstm
ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32WithCifgWithPeepholeNoProjection,
                              LstmLayerFloat32WithCifgWithPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32NoCifgNoPeepholeNoProjection,
                              LstmLayerFloat32NoCifgNoPeepholeNoProjectionTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32NoCifgWithPeepholeWithProjection,
                              LstmLayerFloat32NoCifgWithPeepholeWithProjectionTest)
// Moved to  NeonLayerTests_NDK_Bug.cpp
//ARMNN_AUTO_TEST_CASE_WITH_THF(LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNorm,
//                              LstmLayerFloat32NoCifgWithPeepholeWithProjectionWithLayerNormTest)

// QLstm
ARMNN_AUTO_TEST_CASE_WITH_THF(QLstm, QLstmTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(QLstm1, QLstmTest1)
ARMNN_AUTO_TEST_CASE_WITH_THF(QLstm2, QLstmTest2)

// QuantizedLstm
ARMNN_AUTO_TEST_CASE_WITH_THF(QuantizedLstm, QuantizedLstmTest)

// Mean
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleFloat32, MeanSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleAxisFloat32, MeanSimpleAxisTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanKeepDimsFloat32, MeanKeepDimsTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanMultipleDimsFloat32, MeanMultipleDimsTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts1Float32, MeanVts1Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts2Float32, MeanVts2Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts3Float32, MeanVts3Test<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleQuantisedAsymmS8, MeanSimpleTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleAxisQuantisedAsymmS8, MeanSimpleAxisTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanKeepDimsQuantisedAsymmS8, MeanKeepDimsTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanMultipleDimsQuantisedAsymmS8, MeanMultipleDimsTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts1QuantisedAsymmS8, MeanVts1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts2QuantisedAsymmS8, MeanVts2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts3QuantisedAsymmS8, MeanVts3Test<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleQuantisedAsymm8, MeanSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanSimpleAxisQuantisedAsymm8, MeanSimpleAxisTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanKeepDimsQuantisedAsymm8, MeanKeepDimsTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanMultipleDimsQuantisedAsymm8, MeanMultipleDimsTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts1QuantisedAsymm8, MeanVts1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts2QuantisedAsymm8, MeanVts2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(MeanVts3QuantisedAsymm8, MeanVts3Test<DataType::QAsymmU8>)

// Max
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMaximum, MaximumSimpleTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1Element, MaximumBroadcast1ElementTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1DVector, MaximumBroadcast1DVectorTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumUint8, MaximumUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1ElementUint8, MaximumBroadcast1ElementUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(MaximumBroadcast1DVectorUint8, MaximumBroadcast1DVectorUint8Test)

// Min
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMinimum1, MinimumBroadcast1ElementTest1)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleMinimum2, MinimumBroadcast1ElementTest2)
ARMNN_AUTO_TEST_CASE_WITH_THF(Minimum1DVectorUint8, MinimumBroadcast1DVectorUint8Test)

// Normalization
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleNormalizationAcross, SimpleNormalizationAcrossTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleNormalizationWithin, SimpleNormalizationWithinTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleNormalizationAcrossNhwc, SimpleNormalizationAcrossNhwcTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(AcrossChannelNormalization, AcrossChannelNormalizationTest)

// Resize Bilinear - NCHW data layout
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinear, SimpleResizeBilinearTest<DataType::Float32>, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNop, ResizeBilinearNopTest<DataType::Float32>, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMin, ResizeBilinearSqMinTest<DataType::Float32>, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMin, ResizeBilinearMinTest<DataType::Float32>, DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMag, ResizeBilinearMagTest<DataType::Float32>, DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearUint8,
                              SimpleResizeBilinearTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNopUint8,
                              ResizeBilinearNopTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinUint8,
                              ResizeBilinearSqMinTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinUint8,
                              ResizeBilinearMinTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagUint8,
                              ResizeBilinearMagTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)

ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeBilinear,
                              HalfPixelCentersResizeBilinearTest<DataType::Float32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeBilinear,
                              AlignCornersResizeBilinearTest<DataType::Float32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeBilinearInt8,
                              HalfPixelCentersResizeBilinearTest<DataType::QAsymmS8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeBilinearInt8,
                              AlignCornersResizeBilinearTest<DataType::QAsymmS8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeBilinearUint8,
                              HalfPixelCentersResizeBilinearTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeBilinearUint8,
                              AlignCornersResizeBilinearTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)

// Resize Bilinear - NHWC data layout
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNopNhwc,
                              ResizeBilinearNopTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearNhwc,
                              SimpleResizeBilinearTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinNhwc,
                              ResizeBilinearSqMinTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinNhwc,
                              ResizeBilinearMinTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagNhwc,
                              ResizeBilinearMagTest<DataType::Float32>,
                              DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearNopUint8Nhwc,
                              ResizeBilinearNopTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeBilinearUint8Nhwc,
                              SimpleResizeBilinearTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearSqMinUint8Nhwc,
                              ResizeBilinearSqMinTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMinUint8Nhwc,
                              ResizeBilinearMinTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeBilinearMagUint8Nhwc,
                              ResizeBilinearMagTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeBilinearNhwc,
                              HalfPixelCentersResizeBilinearTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeBilinearNhwc,
                              AlignCornersResizeBilinearTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeBilinearInt8Nhwc,
                              HalfPixelCentersResizeBilinearTest<DataType::QAsymmS8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeBilinearInt8Nhwc,
                              AlignCornersResizeBilinearTest<DataType::QAsymmS8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeBilinearUint8Nhwc,
                              HalfPixelCentersResizeBilinearTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeBilinearUint8Nhwc,
                              AlignCornersResizeBilinearTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)

// Resize NearestNeighbor - NCHW
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighbor,
                              SimpleResizeNearestNeighborTest<DataType::Float32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborNop,
                              ResizeNearestNeighborNopTest<DataType::Float32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMin,
                              ResizeNearestNeighborSqMinTest<DataType::Float32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMin,
                              ResizeNearestNeighborMinTest<DataType::Float32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMag,
                              ResizeNearestNeighborMagTest<DataType::Float32>,
                              DataLayout::NCHW, 0.1f, 50, 0.1f, 50)

ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighborUint8,
                              SimpleResizeNearestNeighborTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborNopUint8,
                              ResizeNearestNeighborNopTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMinUint8,
                              ResizeNearestNeighborSqMinTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMinUint8,
                              ResizeNearestNeighborMinTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMagUint8,
                              ResizeNearestNeighborMagTest<DataType::QAsymmU8>,
                              DataLayout::NCHW, 0.1f, 50, 0.1f, 50)

ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeNearestNeighbour,
                              HalfPixelCentersResizeNearestNeighbourTest<DataType::Float32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeNearestNeighbour,
                              AlignCornersResizeNearestNeighbourTest<DataType::Float32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeNearestNeighbourInt8,
                              HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeNearestNeighbourInt8,
                              AlignCornersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeNearestNeighbourUint8,
                              HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeNearestNeighbourUint8,
                              AlignCornersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                              DataLayout::NCHW)

// Resize NearestNeighbor - NHWC
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborNopNhwc,
                              ResizeNearestNeighborNopTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighborNhwc,
                              SimpleResizeNearestNeighborTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMinNhwc,
                              ResizeNearestNeighborSqMinTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMinNhwc,
                              ResizeNearestNeighborMinTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMagNhwc,
                              ResizeNearestNeighborMagTest<DataType::Float32>,
                              DataLayout::NHWC, 0.1f, 50, 0.1f, 50)

ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborNopUint8Nhwc,
                              ResizeNearestNeighborNopTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleResizeNearestNeighborUint8Nhwc,
                              SimpleResizeNearestNeighborTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborSqMinUint8Nhwc,
                              ResizeNearestNeighborSqMinTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMinUint8Nhwc,
                              ResizeNearestNeighborMinTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(ResizeNearestNeighborMagUint8Nhwc,
                              ResizeNearestNeighborMagTest<DataType::QAsymmU8>,
                              DataLayout::NHWC, 0.1f, 50, 0.1f, 50)

ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeNearestNeighbourNhwc,
                              HalfPixelCentersResizeNearestNeighbourTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeNearestNeighbourNhwc,
                              AlignCornersResizeNearestNeighbourTest<DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeNearestNeighbourInt8Nhwc,
                              HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeNearestNeighbourInt8Nhwc,
                              AlignCornersResizeNearestNeighbourTest<DataType::QAsymmS8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(HalfPixelCentersResizeNearestNeighbourUint8Nhwc,
                              HalfPixelCentersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(AlignCornersResizeNearestNeighbourUint8Nhwc,
                              AlignCornersResizeNearestNeighbourTest<DataType::QAsymmU8>,
                              DataLayout::NHWC)

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

// Strided Slice
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dFloat32, StridedSlice4dFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSlice4dReverseFloat32, StridedSlice4dReverseFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleStrideFloat32, StridedSliceSimpleStrideFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceSimpleRangeMaskFloat32, StridedSliceSimpleRangeMaskFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskFloat32, StridedSliceShrinkAxisMaskFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedSliceShrinkAxisMaskCTSFloat32,
                                          StridedSliceShrinkAxisMaskCTSFloat32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(
    StridedSliceShrinkAxisMaskBitPosition0Dim3Float32, StridedSliceShrinkAxisMaskBitPosition0Dim3Float32Test)
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

// Quantize
ARMNN_AUTO_TEST_CASE_WITH_THF(QuantizeSimpleUint8, QuantizeSimpleUint8Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(QuantizeClampUint8, QuantizeClampUint8Test)

// PReLU
ARMNN_AUTO_TEST_CASE_WITH_THF(PreluFloat32, PreluTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(PreluUint8,   PreluTest<DataType::QAsymmU8>)

// Stack
ARMNN_AUTO_TEST_CASE_WITH_THF(Stack0Axis,           StackAxis0Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput4DAxis1,   StackOutput4DAxis1Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput4DAxis2,   StackOutput4DAxis2Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput4DAxis3,   StackOutput4DAxis3Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput3DInputs3, StackOutput3DInputs3Float32Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(StackOutput5D,        StackOutput5DFloat32Test)

// Transpose
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeFloat32, SimpleTransposeTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeFloat32ValueSet1Test, TransposeValueSet1Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeFloat32ValueSet2Test, TransposeValueSet2Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeFloat32ValueSet3Test, TransposeValueSet3Test<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeQASymms8, SimpleTransposeTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeQASymmS8ValueSet1Test, TransposeValueSet1Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeQASymmS8ValueSet2Test, TransposeValueSet2Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeQASymmS8ValueSet3Test, TransposeValueSet3Test<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeQASymm8, SimpleTransposeTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeQASymm8ValueSet1Test, TransposeValueSet1Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeQASymm8ValueSet2Test, TransposeValueSet2Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeQASymm8ValueSet3Test, TransposeValueSet3Test<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeQSymm16, SimpleTransposeTest<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeQSymm16ValueSet1Test, TransposeValueSet1Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeQSymm16ValueSet2Test, TransposeValueSet2Test<DataType::QSymmS16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(TransposeQSymm16ValueSet3Test, TransposeValueSet3Test<DataType::QSymmS16>)

// TransposeConvolution2d
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dFloatNchw,
                              SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dFloatNhwc,
                              SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              true,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dUint8Nchw,
                              SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleTransposeConvolution2dUint8Nhwc,
                              SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dFloatNchw,
                              SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              false,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dFloatNhwc,
                              SimpleTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              true,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dUint8Nchw,
                              SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedSimpleTransposeConvolution2dUint8Nhwc,
                              SimpleTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dFloatNchw,
                              PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dFloatNhwc,
                              PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              true,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dUint8Nchw,
                              PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(PaddedTransposeConvolution2dUint8Nhwc,
                              PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dFloatNchw,
                              PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              false,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dFloatNhwc,
                              PaddedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              true,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dInt8Nchw,
                              PaddedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dInt8Nhwc,
                              PaddedTransposeConvolution2dTest<DataType::QAsymmS8, DataType::Signed32>,
                              true,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dUint8Nchw,
                              PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedPaddedTransposeConvolution2dUint8Nhwc,
                              PaddedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dFloatNchw,
                              StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dFloatNhwc,
                              StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              true,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dUint8Nchw,
                              StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(StridedTransposeConvolution2dUint8Nhwc,
                              StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dFloatNchw,
                              StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              false,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dFloatNhwc,
                              StridedTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              true,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dUint8Nchw,
                              StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(UnbiasedStridedTransposeConvolution2dUint8Nhwc,
                              StridedTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              true,
                              DataLayout::NHWC)

ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dFloatNchw,
                              MultiChannelTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dFloatNhwc,
                              MultiChannelTransposeConvolution2dTest<DataType::Float32, DataType::Float32>,
                              DataLayout::NHWC)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dUint8Nchw,
                              MultiChannelTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              DataLayout::NCHW)
ARMNN_AUTO_TEST_CASE_WITH_THF(MultiChannelTransposeConvolution2dUint8Nhwc,
                              MultiChannelTransposeConvolution2dTest<DataType::QAsymmU8, DataType::Signed32>,
                              DataLayout::NHWC)

// Abs
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs2d, Abs2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs3d, Abs3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(AbsZero, AbsZeroTest<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(Abs2dSigned32, Abs2dTest<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Abs3dSigned32, Abs3dTest<DataType::Signed32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(AbsZeroSigned32, AbsZeroTest<DataType::Signed32>)

// Rsqrt
ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt2d, Rsqrt2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Rsqrt3d, Rsqrt3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RsqrtZero, RsqrtZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(RsqrtNegative, RsqrtNegativeTest<DataType::Float32>)

// ArgMinMax
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinFloat32, ArgMinSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxFloat32, ArgMaxSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannel, ArgMinChannelTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannel, ArgMaxChannelTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxHeight, ArgMaxHeightTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinWidth, ArgMinWidthTest<DataType::Float32>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinQAsymmS8, ArgMinSimpleTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxQAsymmS8, ArgMaxSimpleTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelQAsymmS8, ArgMinChannelTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelQAsymmS8, ArgMaxChannelTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxHeightQAsymmS8, ArgMaxHeightTest<DataType::QAsymmS8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinWidthQAsymmS8, ArgMinWidthTest<DataType::QAsymmS8>)

ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinQAsymm8, ArgMinSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxQAsymm8, ArgMaxSimpleTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinChannelQAsymm8, ArgMinChannelTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxChannelQAsymm8, ArgMaxChannelTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMaxHeightQAsymm8, ArgMaxHeightTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ArgMinWidthQAsymm8, ArgMinWidthTest<DataType::QAsymmU8>)

// Neg
ARMNN_AUTO_TEST_CASE_WITH_THF(Neg2d, Neg2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Neg3d, Neg3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(NegZero, NegZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(NegNegative, NegNegativeTest<DataType::Float32>)

// Exp
ARMNN_AUTO_TEST_CASE_WITH_THF(Exp2d, Exp2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Exp3d, Exp3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ExpZero, ExpZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ExpNegative, ExpNegativeTest<DataType::Float32>)

// Log
ARMNN_AUTO_TEST_CASE_WITH_THF(Log2d, Log2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Log3d, Log3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogZero, LogZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogNegative, LogNegativeTest<DataType::Float32>)

// Sin
ARMNN_AUTO_TEST_CASE_WITH_THF(Sin2d, Sin2dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(Sin3d, Sin3dTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SinZero, SinZeroTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SinNegative, SinNegativeTest<DataType::Float32>)

// Fill
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFill, SimpleFillTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFillF16, SimpleFillTest<DataType::Float16>)
ARMNN_AUTO_TEST_CASE_WITH_THF(SimpleFillS32, SimpleFillTest<DataType::Signed32>)

// Logical
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalNot, LogicalNotTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalNotInt, LogicalNotIntTest)

ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalAnd, LogicalAndTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalAndInt, LogicalAndIntTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalAndBroadcast1, LogicalAndBroadcast1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalAndBroadcast2, LogicalAndBroadcast2Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalAndBroadcast3, LogicalAndBroadcast3Test)

ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalOr, LogicalOrTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalOrInt, LogicalOrIntTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalOrBroadcast1, LogicalOrBroadcast1Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalOrBroadcast2, LogicalOrBroadcast2Test)
ARMNN_AUTO_TEST_CASE_WITH_THF(LogicalOrBroadcast3, LogicalOrBroadcast3Test)

// ReduceSum
// Moved to NeonLayerTests_NDK_Bug.cpp
//ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceSumFloat32, ReduceSumSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceSumSingleAxisFloat32_1, ReduceSumSingleAxisTest1<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceSumSingleAxisFloat32_2, ReduceSumSingleAxisTest2<DataType::Float32>)
// Moved to NeonLayerTests_NDK_Bug.cpp
//ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceSumSingleAxisFloat32_3, ReduceSumSingleAxisTest3<DataType::Float32>)

// ReduceProd
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceProdSingleAxisFloat32_1, ReduceProdSingleAxisTest1<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceProdSingleAxisFloat32_2, ReduceProdSingleAxisTest2<DataType::Float32>)

// ReduceMax
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceMaxFloat32, ReduceMaxSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceMaxNegativeAxisFloat32, ReduceMaxNegativeAxisTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceMax2Float32, ReduceMaxSimpleTest2<DataType::Float32>)

// ReduceMin
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceMinFloat32, ReduceMinSimpleTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ReduceMinNegativeAxisFloat32, ReduceMinNegativeAxisTest<DataType::Float32>)

// Cast
ARMNN_AUTO_TEST_CASE_WITH_THF(CastInt32ToFloat, CastInt32ToFloat2dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(CastInt8AsymmToFloat, CastInt8AsymmToFloat2dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(CastUIntToFloat, CastUInt8ToFloat2dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(CastFloatToIn8, CastFloat32ToInt82dTest)
ARMNN_AUTO_TEST_CASE_WITH_THF(CastFloatToUInt8, CastFloat32ToUInt82dTest)

// ChannelShuffle
ARMNN_AUTO_TEST_CASE_WITH_THF(ChannelShuffle4DFloat32, ChannelShuffle4DTest<DataType::Float32>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ChannelShuffle4DQAsymmU8, ChannelShuffle4DTest<DataType::QAsymmU8>)
ARMNN_AUTO_TEST_CASE_WITH_THF(ChannelShuffle4DQAsymmS8, ChannelShuffle4DTest<DataType::QAsymmS8>)

#if defined(ARMNNREF_ENABLED)

// The ARMNN_COMPARE_REF_AUTO_TEST_CASE and the ARMNN_COMPARE_REF_FIXTURE_TEST_CASE test units are not available
// if the reference backend is not built

// ============================================================================
// COMPARE tests

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

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareMaxPooling2dWithReference, ComparePooling2dTest, PoolingAlgorithm::Max)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareMaxPooling2dWithReferenceUint8, ComparePooling2dUint8Test,
                                          PoolingAlgorithm::Max)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareAveragePooling2dWithReference, ComparePooling2dTest,
                                          PoolingAlgorithm::Average)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareAveragePooling2dWithReferenceUint8, ComparePooling2dUint8Test,
                                          PoolingAlgorithm::Average)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareL2Pooling2dWithReference, ComparePooling2dTest, PoolingAlgorithm::L2)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(UNSUPPORTED_CompareL2Pooling2dWithReferenceUint8, ComparePooling2dUint8Test,
                                          PoolingAlgorithm::L2)

// Moved to NeonLayerTests_NDK_Bug.cpp
//ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxBeta1WithReference, CompareSoftmaxTest, 1.0f)
//ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxBeta2WithReference, CompareSoftmaxTest, 2.0f)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxUint8Beta1WithReference, CompareSoftmaxUint8Test, 1.0f)
ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareSoftmaxUint8Beta2WithReference, CompareSoftmaxUint8Test, 2.0f)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareAddition, CompareAdditionTest)

ARMNN_COMPARE_REF_AUTO_TEST_CASE_WITH_THF(CompareMultiplicationWithReference, CompareMultiplicationTest)

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
                                    CompareActivationTest, ActivationFunction::SoftReLu, 1u)

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
