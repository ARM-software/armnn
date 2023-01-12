//
// Copyright © 2022-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/test/EndToEndTestImpl.hpp"

#include "backendsCommon/test/AdditionEndToEndTestImpl.hpp"
#include "backendsCommon/test/Convolution2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/ConcatEndToEndTestImpl.hpp"
#include "backendsCommon/test/MultiplicationEndToEndTestImpl.hpp"
#include "backendsCommon/test/Pooling2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/ReshapeEndToEndTestImpl.hpp"
#include "backendsCommon/test/ElementwiseUnaryEndToEndTestImpl.hpp"
#include "backendsCommon/test/SliceEndToEndTestImpl.hpp"
#include "backendsCommon/test/SubtractionEndToEndTestImpl.hpp"
#include "backendsCommon/test/TransposeConvolution2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/TransposeEndToEndTestImpl.hpp"

#include <doctest/doctest.h>

TEST_SUITE("TosaRefEndToEnd")
{
std::vector<BackendId> tosaDefaultBackends = { "TosaRef" };

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

TEST_CASE("TosaRefRsqrtEndtoEndTestFloat32")
{
    ElementwiseUnarySimpleEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends,
                                                             UnaryOperation::Rsqrt);
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