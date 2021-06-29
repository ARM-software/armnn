//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Multiplication")
{
struct MultiplicationFixture : public ParserFlatbuffersFixture
{
    explicit MultiplicationFixture(const std::string & inputShape1,
                                            const std::string & inputShape2,
                                            const std::string & outputShape,
                                            const std::string & activation="NONE")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "MUL" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape1 + R"(,
                            "type": "FLOAT32",
                            "buffer": 0,
                            "name": "inputTensor1",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + inputShape2 + R"(,
                            "type": "FLOAT32",
                            "buffer": 1,
                            "name": "inputTensor2",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + outputShape + R"( ,
                            "type": "FLOAT32",
                            "buffer": 2,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 0, 1 ],
                    "outputs": [ 2 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1 ],
                            "outputs": [ 2 ],
                            "builtin_options_type": "MulOptions",
                            "builtin_options": {
                                "fused_activation_function": )" + activation + R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { }
                ]
            }
        )";
        Setup();
    }
};

struct SimpleMultiplicationFixture : public MultiplicationFixture
{
    SimpleMultiplicationFixture() : MultiplicationFixture("[ 1, 2, 2, 3 ]", "[ 1, 2, 2, 3 ]", "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(SimpleMultiplicationFixture, "ParseMultiplication")
{
    using armnn::DataType;
    RunTest<4, DataType::Float32>(0, {{ "inputTensor1", { 0.0f,  1.0f,  2.0f,
                                                          3.0f,  4.0f,  5.0f,
                                                          6.0f,  7.0f,  8.0f,
                                                          9.0f, 10.0f, 11.0f } },
                                      { "inputTensor2", { 1.0f,  1.0f,  1.0f,
                                                          5.0f,  5.0f,  5.0f,
                                                          1.0f,  1.0f,  1.0f,
                                                          5.0f,  5.0f,  5.0f} } },
                                     {{ "outputTensor", { 0.0f,  1.0f,  2.0f,
                                                         15.0f, 20.0f, 25.0f,
                                                          6.0f,  7.0f,  8.0f,
                                                         45.0f, 50.0f, 55.0f } } });
}

struct MultiplicationBroadcastFixture4D1D : public MultiplicationFixture
{
    MultiplicationBroadcastFixture4D1D() : MultiplicationFixture("[ 1, 2, 2, 3 ]", "[ 1 ]", "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MultiplicationBroadcastFixture4D1D, "ParseMultiplicationBroadcast4D1D")
{
    using armnn::DataType;
    RunTest<4, DataType::Float32>(0, {{ "inputTensor1", { 0.0f,  1.0f,  2.0f,
                                                          3.0f,  4.0f,  5.0f,
                                                          6.0f,  7.0f,  8.0f,
                                                          9.0f, 10.0f, 11.0f } },
                                      { "inputTensor2", { 5.0f } } },
                                     {{ "outputTensor", { 0.0f,  5.0f, 10.0f,
                                                         15.0f, 20.0f, 25.0f,
                                                         30.0f, 35.0f, 40.0f,
                                                         45.0f, 50.0f, 55.0f } } });
}

struct MultiplicationBroadcastFixture1D4D : public MultiplicationFixture
{
    MultiplicationBroadcastFixture1D4D() : MultiplicationFixture("[ 1 ]", "[ 1, 2, 2, 3 ]", "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MultiplicationBroadcastFixture1D4D, "ParseMultiplicationBroadcast1D4D")
{
    using armnn::DataType;
    RunTest<4, DataType::Float32>(0, {{ "inputTensor1", { 3.0f } },
                                      { "inputTensor2", { 0.0f,  1.0f,  2.0f,
                                                          3.0f,  4.0f,  5.0f,
                                                          6.0f,  7.0f,  8.0f,
                                                          9.0f, 10.0f, 11.0f } } },
                                     {{ "outputTensor", { 0.0f,  3.0f,  6.0f,
                                                          9.0f, 12.0f, 15.0f,
                                                         18.0f, 21.0f, 24.0f,
                                                         27.0f, 30.0f, 33.0f } } });
}

}
