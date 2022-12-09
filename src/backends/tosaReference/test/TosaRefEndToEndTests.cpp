//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/test/EndToEndTestImpl.hpp"

#include "backendsCommon/test/AdditionEndToEndTestImpl.hpp"
#include "backendsCommon/test/Convolution2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/Pooling2dEndToEndTestImpl.hpp"
#include "backendsCommon/test/ReshapeEndToEndTestImpl.hpp"
#include "backendsCommon/test/SliceEndToEndTestImpl.hpp"

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

// Conv2d
TEST_CASE("TosaRefConv2dEndtoEndTestFloat32")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC);
}

TEST_CASE("TosaRefConv2dWithoutBiasEndtoEndTestFloat32")
{
    Convolution2dEndToEnd<armnn::DataType::Float32>(tosaDefaultBackends, armnn::DataLayout::NHWC, false);
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

}