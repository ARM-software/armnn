//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//


#include <armnn/Optional.hpp>
#include <armnn/Types.hpp>
#include <tosaReference/TosaRefLayerSupport.hpp>

#include <doctest/doctest.h>

#include <string>

TEST_SUITE("TosaRefLayerSupported")
{

TEST_CASE("IsLayerSupportedTosaReferenceAddition")
{
    armnn::TensorShape shape0 = {1,1,3,4};
    armnn::TensorShape shape1 = {4};
    armnn::TensorShape outShape = {1,1,3,4};
    armnn::TensorInfo in0(shape0, armnn::DataType::Float32);
    armnn::TensorInfo in1(shape1, armnn::DataType::Float32);
    armnn::TensorInfo out(outShape, armnn::DataType::Float32);

    armnn::BaseDescriptor desc;
    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Addition,
                                                     {in0, in1, out},
                                                     desc,
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceAdditionUnsupported")
{
    armnn::TensorShape shape0 = {1,1,3,4};
    armnn::TensorShape shape1 = {4};
    armnn::TensorShape outShape = {1,1,3,4};
    armnn::TensorInfo in0(shape0, armnn::DataType::Signed64);
    armnn::TensorInfo in1(shape1, armnn::DataType::Signed64);
    armnn::TensorInfo out(outShape, armnn::DataType::Signed64);

    armnn::BaseDescriptor desc;
    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Addition,
                                                     {in0, in1, out},
                                                     desc,
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
    REQUIRE(reasonIfNotSupported.find(
        "TOSA Reference Operator: Op_ADD for input: input0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        "TOSA Reference Operator: Op_ADD for input: input1_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        "TOSA Reference Operator: Op_ADD for output: output0_") != std::string::npos);
}

TEST_CASE("IsLayerSupportedTosaReferenceConstant")
{
    armnn::TensorInfo outputInfo({1,1,3,4}, armnn::DataType::Float32);

    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Constant,
                                                     {outputInfo},
                                                     armnn::BaseDescriptor(),
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceConstantUnsupported")
{
    armnn::TensorInfo outputInfo({1,1,3,4}, armnn::DataType::Signed64);

    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Constant,
                                                     {outputInfo},
                                                     armnn::BaseDescriptor(),
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
    REQUIRE(reasonIfNotSupported.find(
            "TOSA Reference Operator: Op_CONST for output: constant_") != std::string::npos);
}

TEST_CASE("IsLayerSupportedTosaReferenceConv2d")
{
    armnn::TensorInfo inputInfo ({ 1, 5, 5, 1 }, armnn::DataType::Float32);
    armnn::TensorInfo outputInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);
    armnn::TensorInfo weightsInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32);
    armnn::TensorInfo biasesInfo ({ 1 }, armnn::DataType::Float32);

    armnn::Convolution2dDescriptor desc;
    desc.m_BiasEnabled = true;

    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Convolution2d,
                                                     {inputInfo, outputInfo, weightsInfo, biasesInfo},
                                                     desc,
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceConv2dUnsupported")
{
    // If inputs and weights are Fp32, output must match.
    armnn::TensorInfo inputInfo ({ 1, 5, 5, 1 }, armnn::DataType::Float32);
    armnn::TensorInfo outputInfo({ 1, 3, 3, 1 }, armnn::DataType::Signed64);
    armnn::TensorInfo weightsInfo({ 1, 3, 3, 1 }, armnn::DataType::Float32, 0.0f, 0, true);
    armnn::TensorInfo biasesInfo ({ 1 }, armnn::DataType::Float32, 0.0f, 0, true);

    armnn::Convolution2dDescriptor desc;
    desc.m_BiasEnabled = true;

    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Convolution2d,
                                                     {inputInfo, outputInfo, weightsInfo, biasesInfo},
                                                     desc,
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
    REQUIRE(reasonIfNotSupported.find(
            "TOSA Reference Operator: Op_CONV2D for input 0: input0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
            "input 1: input1_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
            "and output: output0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
            "has an unsupported input data type combination.") != std::string::npos);
}

TEST_CASE("IsLayerSupportedTosaReferenceMaxPooling2d")
{
    armnn::TensorShape inShape = {1,1,3,4};
    armnn::TensorShape outShape = {1,1,3,4};
    armnn::TensorInfo in(inShape, armnn::DataType::Float32);
    armnn::TensorInfo out(outShape, armnn::DataType::Float32);

    armnn::Pooling2dDescriptor desc;
    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Pooling2d,
                                                     {in, out},
                                                     desc,
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceAvgPooling2d_IgnoreValue")
{
    armnn::TensorShape inShape = {1,1,3,4};
    armnn::TensorShape outShape = {1,1,3,4};
    armnn::TensorInfo in(inShape, armnn::DataType::Float32);
    armnn::TensorInfo out(outShape, armnn::DataType::Float32);

    armnn::Pooling2dDescriptor desc;
    desc.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
    desc.m_PoolType = armnn::PoolingAlgorithm::Average;

    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Pooling2d,
                                                     {in, out},
                                                     desc,
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceAvgPooling2d_InputOutputDatatypeDifferent")
{
    armnn::TensorShape inShape = {1,1,3,4};
    armnn::TensorShape outShape = {1,1,3,4};
    armnn::TensorInfo in(inShape, armnn::DataType::QAsymmS8);
    armnn::TensorInfo out(outShape, armnn::DataType::Signed32);

    armnn::Pooling2dDescriptor desc;
    desc.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
    desc.m_PoolType = armnn::PoolingAlgorithm::Average;

    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Pooling2d,
                                                     {in, out},
                                                     desc,
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(supported);
}

TEST_CASE("IsLayerSupportedTosaReferenceMaxPooling2dUnsupported")
{
    armnn::TensorShape inShape = {1,1,3,4};
    armnn::TensorShape outShape = {1,1,3,4};
    armnn::TensorInfo in(inShape, armnn::DataType::Signed64);
    armnn::TensorInfo out(outShape, armnn::DataType::Signed64);

    armnn::Pooling2dDescriptor desc;
    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Pooling2d,
                                                     {in, out},
                                                     desc,
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
    REQUIRE(reasonIfNotSupported.find(
        "TOSA Reference Operator: Op_MAX_POOL2D for input: input0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        "TOSA Reference Operator: Op_MAX_POOL2D for output: output0_") != std::string::npos);
}

TEST_CASE("IsLayerSupportedTosaReferenceAvgPooling2dUnsupported_InputOutputDatatypeDifferent")
{
    armnn::TensorShape inShape = {1,1,3,4};
    armnn::TensorShape outShape = {1,1,3,4};
    armnn::TensorInfo in(inShape, armnn::DataType::Float32);
    armnn::TensorInfo out(outShape, armnn::DataType::Float16);

    armnn::Pooling2dDescriptor desc;
    desc.m_PaddingMethod = armnn::PaddingMethod::IgnoreValue;
    desc.m_PoolType = armnn::PoolingAlgorithm::Average;

    armnn::TosaRefLayerSupport supportChecker;
    std::string reasonIfNotSupported;
    auto supported = supportChecker.IsLayerSupported(armnn::LayerType::Pooling2d,
                                                     {in, out},
                                                     desc,
                                                     armnn::EmptyOptional(),
                                                     armnn::EmptyOptional(),
                                                     reasonIfNotSupported);

    CHECK(!supported);
    REQUIRE(reasonIfNotSupported.find(
        "TOSA Reference Operator: Op_AVG_POOL2D for input: intermediate0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        " and output: output0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        " has an unsupported input data type: 8 to output data type: 10") != std::string::npos);
}

}
