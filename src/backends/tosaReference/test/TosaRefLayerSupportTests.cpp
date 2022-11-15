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
        "TOSA Reference Operator: Op_ADD for input: Op_ADD_input0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        "TOSA Reference Operator: Op_ADD for input: Op_ADD_input1_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        "TOSA Reference Operator: Op_ADD for output: Op_ADD_output0_") != std::string::npos);
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
        "TOSA Reference Operator: Op_MAX_POOL2D for input: Op_MAX_POOL2D_input0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        "TOSA Reference Operator: Op_MAX_POOL2D for output: Op_MAX_POOL2D_output0_") != std::string::npos);
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
        "TOSA Reference Operator: Op_AVG_POOL2D for input: Op_PAD_intermediate0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        " and output: Op_AVG_POOL2D_output0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find(
        " has an unsupported input data type: 8 to output data type: 10") != std::string::npos);
}

}
