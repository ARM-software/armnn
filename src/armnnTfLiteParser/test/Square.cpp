//
// Copyright © 2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_Square")
{
struct SquareFixture : public ParserFlatbuffersFixture
{
    explicit SquareFixture(const std::string & inputShape,
                           const std::string & outputShape)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "SQUARE" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": "FLOAT32",
                            "buffer": 0,
                            "name": "inputTensor",
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
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0 ],
                            "outputs": [ 1 ],
                            "builtin_options_type": "SquareOptions",
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

struct SimpleSquareFixture : public SquareFixture
{
    SimpleSquareFixture() : SquareFixture("[ 1, 2, 2, 3 ]", "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(SimpleSquareFixture, "ParseSquare")
{
    using armnn::DataType;
    RunTest<4, DataType::Float32>(0, {{ "inputTensor", { 0.0f,  1.0f,  2.0f,
                                                         3.0f,  4.0f,  5.0f,
                                                         6.0f,  7.0f,  8.0f,
                                                         9.0f, 10.0f, 11.0f }}} ,
                                     {{ "outputTensor", { 0.0f,    1.0f,   4.0f,
                                                          9.0f,   16.0f,  25.0f,
                                                          36.0f,  49.0f,  64.0f,
                                                          81.0f, 100.0f, 121.0f }}} );
}

}
