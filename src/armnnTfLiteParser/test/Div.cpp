//
// Copyright © 2020 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Div")
{
struct DivFixture : public ParserFlatbuffersFixture
{
    explicit DivFixture(const std::string & inputShape1,
                        const std::string & inputShape2,
                        const std::string & outputShape,
                        const std::string & activation="NONE")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "DIV" } ],
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
                            "builtin_options_type": "DivOptions",
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

struct SimpleDivFixture : public DivFixture
{
    SimpleDivFixture() : DivFixture("[ 1, 2, 2, 3 ]", "[ 1, 2, 2, 3 ]", "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(SimpleDivFixture, "ParseDiv")
{
    using armnn::DataType;
    float Inf = std::numeric_limits<float>::infinity();
    float NaN = std::numeric_limits<float>::quiet_NaN();

    RunTest<4, DataType::Float32>(0, {{ "inputTensor1", { 0.0f,  1.0f,  2.0f,
                                                          3.0f,  4.0f,  5.0f,
                                                          6.0f,  7.0f,  8.0f,
                                                          9.0f, 10.0f, -11.0f } },
                                      { "inputTensor2", { 0.0f,  0.0f,  4.0f,
                                                          3.0f,  40.0f,  5.0f,
                                                          6.0f,  7.0f,  8.0f,
                                                          9.0f,  10.0f,  11.0f} } },
                                     {{ "outputTensor", { NaN,   Inf,  0.5f,
                                                          1.0f,  0.1f, 1.0f,
                                                          1.0f,  1.0f, 1.0f,
                                                          1.0f,  1.0f, -1.0f } } });
}


struct DynamicDivFixture : public DivFixture
{
    DynamicDivFixture() : DivFixture("[ 1, 2, 2, 3 ]", "[ 1, 2, 2, 3 ]", "[  ]") {}
};

TEST_CASE_FIXTURE(DynamicDivFixture, "ParseDynamicDiv")
{
    using armnn::DataType;
    float Inf = std::numeric_limits<float>::infinity();
    float NaN = std::numeric_limits<float>::quiet_NaN();

    RunTest<4, DataType::Float32, DataType::Float32>(0, {{ "inputTensor1", { 0.0f,  1.0f,  2.0f,
                                                            3.0f,  4.0f,  5.0f,
                                                            6.0f,  7.0f,  8.0f,
                                                            9.0f, 10.0f, -11.0f } },
                                      { "inputTensor2", { 0.0f,  0.0f,  4.0f,
                                                            3.0f,  40.0f,  5.0f,
                                                            6.0f,  7.0f,  8.0f,
                                                            9.0f,  10.0f,  11.0f} } },
                                  {{ "outputTensor", { NaN,   Inf,  0.5f,
                                                         1.0f,  0.1f, 1.0f,
                                                         1.0f,  1.0f, 1.0f,
                                                         1.0f,  1.0f, -1.0f } } }, true);
}

}
