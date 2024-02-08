//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/test/EndToEndTestImpl.hpp"

#include "backendsCommon/test/BatchMatMulEndToEndTestImpl.hpp"
#include "backendsCommon/test/Convolution2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/layerTests/CastTestImpl.hpp"
#include "backendsCommon/test/DepthwiseConvolution2dEndToEndTests.hpp"
#include "backendsCommon/test/ElementwiseBinaryEndToEndTestImpl.hpp"
#include "backendsCommon/test/Pooling2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/ResizeEndToEndTestImpl.hpp"

#include <doctest/doctest.h>

TEST_SUITE("GpuFsaEndToEnd")
{

std::vector<BackendId> gpuFsaDefaultBackends = {"GpuFsa"};

// BatchMatMul
TEST_CASE("RefBatchMatMulEndToEndFloat32Test")
{
    BatchMatMulEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends);
}

// Cast
TEST_CASE("GpuFsaCastEndtoEndTestFloat32ToFloat16")
{
    using namespace half_float::literal;

    std::vector<unsigned int> inputShape { 2, 2, 2 };

    std::vector<float> inputValues { -3.5f, -1.2f, -8.6f, -2.0f, -1.5f, -1.3f, -0.5f, -0.4f };

    std::vector<armnn::Half> outputValues { -3.50_h, -1.20_h, -8.6_h, -2._h, -1.50_h, -1.30_h, -0.50_h, -0.40_h };

    CastSimpleTest<DataType::Float32, DataType::Float16, float, armnn::Half>(gpuFsaDefaultBackends,
                                                                             inputShape,
                                                                             inputValues,
                                                                             outputValues,
                                                                             1.0f,
                                                                             0);
}

// Conv2d
TEST_CASE("GpuFsaConv2dEndtoEndTestFloat32")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("GpuFsaConv2dWithoutBiasEndtoEndTestFloat32")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends, armnn::DataLayout::NHWC, false);
}

TEST_CASE("GpuFsaDepthwiseConvolution2dEndtoEndTestFloat32")
{
    DepthwiseConvolution2dEndToEnd<armnn::DataType::Float32, armnn::DataType::Float32>(gpuFsaDefaultBackends,
                                                                                       armnn::DataLayout::NHWC);
}

// ElementwiseBinary Add
TEST_CASE("GpuFsaElementwiseBinaryAddTestFloat32")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends, BinaryOperation::Add);
}

TEST_CASE("GpuFsaElementwiseBinaryAddTestFloat16")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float16>(gpuFsaDefaultBackends, BinaryOperation::Add);
}

// ElementwiseBinary Mul
TEST_CASE("GpuFsaElementwiseBinaryMulTestFloat32")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends, BinaryOperation::Mul);
}

TEST_CASE("GpuFsaElementwiseBinaryMulTestFloat16")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float16>(gpuFsaDefaultBackends, BinaryOperation::Mul);
}

// ElementwiseBinary Sub
TEST_CASE("GpuFsaElementwiseBinarySubTestFloat32")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends, BinaryOperation::Sub);
}

TEST_CASE("GpuFsaElementwiseBinarySubTestFloat16")
{
    ElementwiseBinarySimple3DEndToEnd<armnn::DataType::Float16>(gpuFsaDefaultBackends, BinaryOperation::Sub);
}

// Pooling 2D
// Average Pool 2D
TEST_CASE("GpuFsaAvgPool2DEndtoEndTestFloat32")
{
    AvgPool2dEndToEnd<DataType::Float32>(gpuFsaDefaultBackends);
}

TEST_CASE("GpuFsaAvgPool2DEndtoEndTestFloat16")
{

    AvgPool2dEndToEndFloat16<DataType::Float16>(gpuFsaDefaultBackends);
}

TEST_CASE("UNSUPPORTED_GpuFsaAvgPool2DIgnoreValueEndtoEndTestFloat32")
{
    // Exclude padding must be set to true in Attributes! to be supported by GPU
    try
    {
        AvgPool2dEndToEnd<DataType::Float32>(gpuFsaDefaultBackends, PaddingMethod::IgnoreValue);
        FAIL("An exception should have been thrown");
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        CHECK(strcmp(e.what(), "Failed to assign a backend to each layer") == 0);
    }
}

// Max Pool 2D
TEST_CASE("GpuFsaMaxPool2DEndtoEndTestFloat32")
{
    MaxPool2dEndToEnd<DataType::Float32>(gpuFsaDefaultBackends);
}

TEST_CASE("GpuFsaMaxPool2DEndtoEndTestFloat16")
{
    MaxPool2dEndToEndFloat16<DataType::Float16>(gpuFsaDefaultBackends);
}

TEST_CASE("UNSUPPORTED_GpuFsaMaxPool2DIgnoreValueEndtoEndTestFloat32")
{
    // Exclude padding must be set to true in Attributes! to be supported by GPU
    try
    {
        MaxPool2dEndToEnd<DataType::Float32>(gpuFsaDefaultBackends, PaddingMethod::IgnoreValue);
        FAIL("An exception should have been thrown");
    }
    catch (const armnn::InvalidArgumentException& e)
    {
        CHECK(strcmp(e.what(), "Failed to assign a backend to each layer") == 0);
    }
}

// Resize Bilinear
TEST_CASE("GpuFsaResizeBilinearEndToEndFloatNhwcTest")
{
    ResizeBilinearEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends, armnn::DataLayout::NHWC);
}

// Resize NearestNeighbor
TEST_CASE("GpuFsaResizeNearestNeighborEndToEndFloatNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("GpuFsaResizeNearestNeighborEndToEndFloatAlignCornersNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends, armnn::DataLayout::NHWC, 
                                                            true, false);
}

TEST_CASE("GpuFsaResizeNearestNeighborEndToEndFloatHalfPixelNhwcTest")
{
    ResizeNearestNeighborEndToEnd<armnn::DataType::Float32>(gpuFsaDefaultBackends, armnn::DataLayout::NHWC, 
                                                            false, true);
}

}
