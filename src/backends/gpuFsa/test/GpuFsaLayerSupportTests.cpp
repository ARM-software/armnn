//
// Copyright © 2022-2024 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <armnn/Optional.hpp>
#include <armnn/Types.hpp>

#include <gpuFsa/GpuFsaLayerSupport.hpp>

#include <doctest/doctest.h>

#include <iostream>

using namespace armnn;

TEST_SUITE("GpuFsaLayerSupport")
{

TEST_CASE("IsLayerSupportedCast")
{
    armnn::TensorInfo inputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float32);
    armnn::TensorInfo outputTensorInfo({1, 3, 2, 3}, armnn::DataType::Float16);

    BaseDescriptor desc;

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Cast,
                                                     {inputTensorInfo, outputTensorInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);
    CHECK(supported);
}

TEST_CASE("IsLayerSupportedGpuFsaConv2d")
{
    TensorInfo inputInfo ({ 1, 5, 5, 1 }, DataType::Float32);
    TensorInfo outputInfo({ 1, 3, 3, 1 }, DataType::Float32);
    TensorInfo weightsInfo({ 1, 3, 3, 1 }, DataType::Float32, 0.0f, 0, true);
    TensorInfo biasesInfo ({ 1 }, DataType::Float32, 0.0f, 0, true);

    Convolution2dDescriptor desc;
    desc.m_BiasEnabled = true;
    desc.m_DataLayout = DataLayout::NHWC;

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Convolution2d,
                                                     {inputInfo, outputInfo, weightsInfo, biasesInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);
    CHECK(supported);
}

TEST_CASE("IsLayerSupportedGpuFsaConv2dUnsupported")
{
    TensorInfo inputInfo ({ 1, 5, 5, 1 }, DataType::Float32);
    TensorInfo outputInfo({ 1, 3, 3, 1 }, DataType::Float32);
    TensorInfo weightsInfo({ 1, 3, 3, 1 }, DataType::Float32, 0.0f, 0, true);

    // NCHW is unsupported.
    Convolution2dDescriptor desc;
    desc.m_DataLayout = DataLayout::NCHW;

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Convolution2d,
                                                     {inputInfo, outputInfo, weightsInfo, TensorInfo()},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);
    CHECK(!supported);
    REQUIRE(reasonIfNotSupported.find("NCHW not supported by this kernel") != std::string::npos);
}

TEST_CASE("IsLayerSupportedGpuFsaElementWiseBinaryAdd")
{
    TensorInfo input0Info({ 2, 2 }, DataType::Float32);
    TensorInfo input1Info({ 2, 2 }, DataType::Float32);
    TensorInfo outputInfo({ 2, 2 }, DataType::Float32);

    ElementwiseBinaryDescriptor desc;
    desc.m_Operation = BinaryOperation::Add;

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::ElementwiseBinary,
                                                     {input0Info, input1Info, outputInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);
    CHECK(supported);
}

TEST_CASE("IsLayerSupportedGpuFsaElementWiseBinarySub")
{
    TensorInfo input0Info({ 2, 2 }, DataType::Float32);
    TensorInfo input1Info({ 2, 2 }, DataType::Float32);
    TensorInfo outputInfo({ 2, 2 }, DataType::Float32);

    ElementwiseBinaryDescriptor desc;
    desc.m_Operation = BinaryOperation::Sub;

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::ElementwiseBinary,
                                                     {input0Info, input1Info, outputInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);
    CHECK(supported);
}

TEST_CASE("IsLayerSupportedGpuFsaPooling2d")
{
    TensorInfo inputInfo({ 1, 3, 4, 1 }, DataType::Float32);
    TensorInfo outputInfo({ 1, 2, 2, 1 }, DataType::Float32);

    Pooling2dDescriptor desc{};
    desc.m_PoolType = PoolingAlgorithm::Max;
    desc.m_PadLeft    = 0;
    desc.m_PadRight   = 0;
    desc.m_PadTop     = 0;
    desc.m_PadBottom  = 0;
    desc.m_PoolWidth  = 2;
    desc.m_PoolHeight = 2;
    desc.m_StrideX    = 1;
    desc.m_StrideY    = 1;
    desc.m_OutputShapeRounding = OutputShapeRounding::Floor;
    desc.m_PaddingMethod = PaddingMethod::Exclude;
    desc.m_DataLayout  = DataLayout::NHWC;

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Pooling2d,
                                                     {inputInfo, outputInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);
    CHECK(supported);
}

}