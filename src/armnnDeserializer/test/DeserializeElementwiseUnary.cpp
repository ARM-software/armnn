//
// Copyright © 2021,2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE(Deserializer)
{
struct ElementwiseUnaryFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ElementwiseUnaryFixture(const std::string& inputShape,
                                     const std::string& outputShape,
                                     const std::string& dataType,
                                     const std::string& unaryOperation = "Abs")
    {
        m_JsonString = R"(
            {
                inputIds: [0],
                outputIds: [2],
                layers: [
                    {
                        layer_type: "InputLayer",
                        layer: {
                            base: {
                                layerBindingId: 0,
                                base: {
                                    index: 0,
                                    layerName: "InputLayer",
                                    layerType: "Input",
                                    inputSlots: [{
                                        index: 0,
                                        connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + inputShape + R"(,
                                            dataType: )" + dataType + R"(
                                        }
                                    }]
                                }
                            }
                        }
                    },
                    {
                        layer_type: "ElementwiseUnaryLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "ElementwiseUnaryLayer",
                                layerType: "ElementwiseUnary",
                                inputSlots: [{
                                    index: 0,
                                    connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                }],
                                outputSlots: [{
                                    index: 0,
                                    tensorInfo: {
                                        dimensions: )" + outputShape + R"(,
                                        dataType: )" + dataType + R"(
                                    }
                                }]
                            },
                            descriptor: {
                                activationFunction: )" + unaryOperation + R"(
                            },
                        }
                    },
                    {
                        layer_type: "OutputLayer",
                        layer: {
                            base:{
                                layerBindingId: 2,
                                base: {
                                    index: 2,
                                    layerName: "OutputLayer",
                                    layerType: "Output",
                                    inputSlots: [{
                                        index: 0,
                                        connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + outputShape + R"(,
                                            dataType: )" + dataType + R"(
                                        },
                                    }],
                                }
                            }
                        },
                    }
                ]
            }
        )";
        Setup();
    }
};

struct SimpleAbsFixture : ElementwiseUnaryFixture
{
    SimpleAbsFixture() : ElementwiseUnaryFixture("[ 1, 2, 2, 2 ]", // inputShape
                                                 "[ 1, 2, 2, 2 ]", // outputShape
                                                 "Float32",        // dataType
                                                 "Abs")            // unaryOperation
    {}
};

FIXTURE_TEST_CASE(SimpleAbsTest, SimpleAbsFixture)
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        {{"InputLayer", {-100.0f, -50.5f, -25.9999f, -0.5f, 0.0f, 1.5555f, 25.5f, 100.0f}}},
        {{"OutputLayer", {100.0f, 50.5f, 25.9999f, 0.5f, 0.0f, 1.5555f, 25.5f, 100.0f}}});
}

struct SimpleLogFixture : ElementwiseUnaryFixture
{
    SimpleLogFixture() : ElementwiseUnaryFixture("[ 1, 2, 2, 2 ]", // inputShape
                                                 "[ 1, 2, 2, 2 ]", // outputShape
                                                 "Float32",        // dataType
                                                 "Log")            // unaryOperation
    {}
};

FIXTURE_TEST_CASE(SimpleLogTest, SimpleLogFixture)
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        {{"InputLayer", {1.0f, 2.1f, 3.2f, 4.3f, 10.f, 100.f, 25.5f, 200.0f}}},
        {{"OutputLayer", {0.f, 0.74193734472f, 1.16315080981f, 1.4586150227f,
                                                2.30258509299f, 4.60517018599f, 3.23867845216f, 5.29831736655f}}});
}

struct SimpleNegFixture : ElementwiseUnaryFixture
{
    SimpleNegFixture() : ElementwiseUnaryFixture("[ 1, 2, 2, 2 ]", // inputShape
                                                 "[ 1, 2, 2, 2 ]", // outputShape
                                                 "Float32",        // dataType
                                                 "Neg")            // unaryOperation
    {}
};

FIXTURE_TEST_CASE(SimpleNegTest, SimpleNegFixture)
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        {{"InputLayer", {100.0f, 50.5f, 25.9999f, 0.5f, 0.0f, -1.5555f, -25.5f, -100.0f}}},
        {{"OutputLayer", {-100.0f, -50.5f, -25.9999f, -0.5f, 0.0f, 1.5555f, 25.5f, 100.0f}}});
}

struct SimpleSinFixture : ElementwiseUnaryFixture
{
    SimpleSinFixture() : ElementwiseUnaryFixture("[ 1, 2, 2, 2 ]", // inputShape
                                                 "[ 1, 2, 2, 2 ]", // outputShape
                                                 "Float32",        // dataType
                                                 "Sin")            // unaryOperation
    {}
};

FIXTURE_TEST_CASE(SimpleSinTest, SimpleSinFixture)
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        {{"InputLayer", {-100.0f, -50.5f, -25.9999f, -0.5f, 0.0f, 1.5555f, 25.5f, 100.0f}}},
        {{"OutputLayer", {0.50636564111f, -0.23237376165f, -0.76249375473f, -0.4794255386f,
                                                0.0f, 0.99988301347f, 0.35905835402f, -0.50636564111f}}});
}

struct SimpleCeilFixture : ElementwiseUnaryFixture
{
    SimpleCeilFixture() : ElementwiseUnaryFixture("[ 1, 2, 2, 2 ]", // inputShape
                                                  "[ 1, 2, 2, 2 ]", // outputShape
                                                  "Float32",        // dataType
                                                  "Ceil")           // unaryOperation
    {}
};

FIXTURE_TEST_CASE(SimpleCeilTest, SimpleCeilFixture)
{
    RunTest<4, armnn::DataType::Float32>(
            0,
            {{"InputLayer", {-100.0f, -50.5f, -25.9999f, -0.5f, 0.0f, 1.5555f, 25.5f, 100.0f}}},
            {{"OutputLayer", {-100.0f, -50.0f, -25.0f, 0.0f, 0.0f, 2.0f, 26.0f, 100.0f}}});
}
}