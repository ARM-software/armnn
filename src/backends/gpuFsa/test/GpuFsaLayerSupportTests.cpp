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

TEST_CASE("IsLayerSupportedGpuFsaActivation")
{
    TensorInfo inputInfo ({ 1, 5, 5, 1 }, DataType::Float32);
    TensorInfo outputInfo({ 1, 5, 5, 1 }, DataType::Float32);

    ActivationDescriptor desc{};

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Activation,
                                                     {inputInfo, outputInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);
    CHECK(supported);
}

TEST_CASE("IsLayerSupportedGpuFsaBatchMatMul")
{
    TensorInfo input0Info({ 2, 2 }, DataType::Float32);
    TensorInfo input1Info({ 2, 2 }, DataType::Float32);
    TensorInfo outputInfo({ 2, 2 }, DataType::Float32);

    BatchMatMulDescriptor desc{};

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::BatchMatMul,
                                                     {input0Info, input1Info, outputInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);
    CHECK(supported);
}

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

TEST_CASE("IsLayerSupportedGpuFsaElementWiseBinary")
{
    TensorInfo input0Info({ 2, 2 }, DataType::Float32);
    TensorInfo input1Info({ 2, 2 }, DataType::Float32);
    TensorInfo outputInfo({ 2, 2 }, DataType::Float32);

    ElementwiseBinaryDescriptor desc;
    SUBCASE("Add")
    {
        desc.m_Operation = BinaryOperation::Add;
    }
    SUBCASE("Mul")
    {
        desc.m_Operation = BinaryOperation::Mul;
    }
    SUBCASE("Sub")
    {
        desc.m_Operation = BinaryOperation::Sub;
    }

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

TEST_CASE("UNSUPPORTED_IsLayerSupportedGpuFsaReshape")
{
    TensorInfo inputInfo{};
    TensorInfo outputInfo{};

    SUBCASE("Float32")
    {
        inputInfo  = { { 2, 3 }, DataType::Float32 };
        outputInfo = { { 6 }   , DataType::Float32 };
    }

    SUBCASE("Float16")
    {
        inputInfo  = { { 2, 3 }, DataType::Float16 };
        outputInfo = { { 6 }   , DataType::Float16 };
    }

    SUBCASE("Int32")
    {
        inputInfo  = { { 2, 3 }, DataType::Signed32 };
        outputInfo = { { 6 }   , DataType::Signed32 };
    }

    SUBCASE("Int16")
    {
        inputInfo  = { { 2, 3 }, DataType::QSymmS16 };
        outputInfo = { { 6 }   , DataType::QSymmS16 };
    }

    SUBCASE("UInt8")
    {
        inputInfo  = { { 2, 3 }, DataType::QAsymmU8 };
        outputInfo = { { 6 }   , DataType::QAsymmU8 };
    }

    SUBCASE("Int8")
    {
        inputInfo  = { { 2, 3 }, DataType::QAsymmS8 };
        outputInfo = { { 6 }   , DataType::QAsymmS8 };
    }

    ReshapeDescriptor desc;
    desc.m_TargetShape = outputInfo.GetShape();

    GpuFsaLayerSupport supportChecker;
    std::string        reasonIfNotSupported;

    auto supported = supportChecker.IsLayerSupported(LayerType::Reshape,
                                                     { inputInfo, outputInfo },
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
}

TEST_CASE("IsLayerSupportedGpuFsaResize")
{
    TensorInfo inputInfo({ 1, 5, 5, 1 }, DataType::Float32);
    TensorInfo outputInfo({ 1, 10, 10, 1 }, DataType::Float32);

    ResizeDescriptor desc{};
    desc.m_Method = ResizeMethod::NearestNeighbor;
    desc.m_TargetHeight = 10;
    desc.m_TargetWidth = 10;
    desc.m_DataLayout = DataLayout::NHWC;

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Resize,
                                                     {inputInfo, outputInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);
    CHECK(supported);
}

TEST_CASE("UNSUPPORTED_IsLayerSupportedGpuFsaSoftmax")
{
    TensorInfo inputInfo({ 2, 2 }, DataType::Float32);
    TensorInfo outputInfo({ 2, 2 }, DataType::Float32);

    SoftmaxDescriptor desc;
    desc.m_Axis = 1;
    desc.m_Beta = 1.0f;

    GpuFsaLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(LayerType::Softmax,
                                                     {inputInfo, outputInfo},
                                                     desc,
                                                     EmptyOptional(),
                                                     EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
}

}