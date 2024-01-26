//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "backendsCommon/test/EndToEndTestImpl.hpp"

#include "backendsCommon/test/Convolution2dEndToEndTestImpl.hpp"

#include "backendsCommon/test/DepthwiseConvolution2dEndToEndTests.hpp"
#include "backendsCommon/test/ElementwiseBinaryEndToEndTestImpl.hpp"

#include <doctest/doctest.h>

TEST_SUITE("GpuFsaEndToEnd")
{

std::vector<BackendId> gpuFsaDefaultBackends = {"GpuFsa"};

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

}
