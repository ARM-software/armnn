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
    REQUIRE(reasonIfNotSupported.find("TOSA Reference addition: Op_ADD_input0_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find("TOSA Reference addition: Op_ADD_input1_") != std::string::npos);
    REQUIRE(reasonIfNotSupported.find("TOSA Reference addition: Op_ADD_output0_") != std::string::npos);
}

}
