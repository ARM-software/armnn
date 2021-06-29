//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <doctest/doctest.h>
#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Prelu")
{
struct PreluFixture : public ParserFlatbuffersFixture
{
    explicit PreluFixture(const std::string& inputShape,
                          const std::string& alphaShape,
                          const std::string& outputShape,
                          const std::string& inputIndex,
                          const std::string& alphaData)
    {
        m_JsonString = R"(
            {
              "version": 3,
              "operator_codes": [
                {
                  "builtin_code": "PRELU",
                  "version": 1
                }
              ],
              "subgraphs": [
                {
                  "tensors": [
                    {
                      "shape": )" + inputShape + R"(,
                      "type": "FLOAT32",
                      "buffer": 1,
                      "name": "input0",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      },
                      "is_variable": false
                    },
                    {
                      "shape": )" + alphaShape + R"(,
                      "type": "FLOAT32",
                      "buffer": 2,
                      "name": "input1",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      },
                      "is_variable": false
                    },
                    {
                      "shape": )" + outputShape + R"(,
                      "type": "FLOAT32",
                      "buffer": 3,
                      "name": "output",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      },
                      "is_variable": false
                    }
                  ],
                  "inputs": )" + inputIndex + R"(,
                  "outputs": [
                    2
                  ],
                  "operators": [
                    {
                      "opcode_index": 0,
                      "inputs": [
                        0,
                        1
                      ],
                      "outputs": [
                        2
                      ],
                      "builtin_options_type": "NONE",
                      "custom_options_format": "FLEXBUFFERS"
                    }
                  ],
                  "name": "main"
                }
              ],
              "description": "MLIR Converted.",
              "buffers": [
                {
                },
                {
                },
                { )" + alphaData + R"(
                },
                {
                }
              ]
            }
        )";
        Setup();
    }
};

struct PreluNetworkFixture : public ParserFlatbuffersFixture
{
    explicit PreluNetworkFixture()
    {
        m_JsonString = R"(
            {
              "version": 3,
              "operator_codes": [
                {
                  "builtin_code": "PRELU",
                  "version": 1
                },
                {
                  "builtin_code": "MUL",
                  "version": 1
                },
                {
                  "builtin_code": "ADD",
                  "version": 1
                }
              ],
              "subgraphs": [
                {
                  "tensors": [
                    {
                      "shape": [
                        1,
                        2,
                        3
                      ],
                      "type": "FLOAT32",
                      "buffer": 6,
                      "name": "output",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      },
                    },
                    {
                      "shape": [
                        1,
                        2,
                        3
                      ],
                      "type": "FLOAT32",
                      "buffer": 5,
                      "name": "mul",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      }
                    },
                    {
                      "shape": [
                        1,
                        2,
                        3
                      ],
                      "type": "FLOAT32",
                      "buffer": 1,
                      "name": "input0",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      }
                    },
                    {
                      "shape": [
                        2,
                        3
                      ],
                      "type": "FLOAT32",
                      "buffer": 2,
                      "name": "alpha",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      }
                    },
                    {
                      "shape": [
                        1
                      ],
                      "type": "FLOAT32",
                      "buffer": 3,
                      "name": "const0",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      }
                    },
                    {
                      "shape": [
                        1,
                        2,
                        3
                      ],
                      "type": "FLOAT32",
                      "buffer": 4,
                      "name": "prelumul",
                      "quantization": {
                        "details_type": "NONE",
                        "quantized_dimension": 0
                      }
                    }
                  ],
                  "inputs": [
                    2
                  ],
                  "outputs": [
                    0
                  ],
                  "operators": [
                    {
                      "opcode_index": 0,
                      "inputs": [
                        2,
                        3
                      ],
                      "outputs": [
                        5
                      ],
                      "builtin_options_type": "NONE",
                      "custom_options_format": "FLEXBUFFERS"
                    },
                    {
                      "opcode_index": 1,
                      "inputs": [
                        5,
                        4
                      ],
                      "outputs": [
                        1
                      ],
                      "builtin_options_type": "MulOptions",
                      "builtin_options": {
                        "fused_activation_function": "NONE"
                      },
                      "custom_options_format": "FLEXBUFFERS"
                    },
                    {
                      "opcode_index": 2,
                      "inputs": [
                        5,
                        1
                      ],
                      "outputs": [
                        0
                      ],
                      "builtin_options_type": "AddOptions",
                      "builtin_options": {
                        "fused_activation_function": "NONE"
                      },
                      "custom_options_format": "FLEXBUFFERS"
                    }
                  ],
                  "name": "main"
                }
              ],
              "buffers": [
                {
                },
                {
                },
                {
                  "data": [
                    0,
                    0,
                    128,
                    62,
                    0,
                    0,
                    128,
                    62,
                    0,
                    0,
                    128,
                    62,
                    0,
                    0,
                    128,
                    62,
                    0,
                    0,
                    128,
                    62,
                    0,
                    0,
                    128,
                    62
                  ]
                },
                {
                  "data": [
                    0,
                    0,
                    160,
                    64
                  ]
                },
                {
                },
                {
                },
                {
                },
                {
                }
              ],
            }
        )";
        Setup();
    }
};

struct SimplePreluFixture : PreluFixture
{
    SimplePreluFixture() : PreluFixture("[ 2, 3 ]",
                                        "[ 1 ]",
                                        "[ 2, 3 ]",
                                        "[ 0, 1 ]",
                                        "") {}
};

struct PreluConstAlphaFixture : PreluFixture
{
    PreluConstAlphaFixture() : PreluFixture(
        "[ 1, 2, 3 ]",
        "[ 1, 2, 3 ]",
        "[ 1, 2, 3 ]",
        "[ 0 ]",
        "\"data\": [ 0, 0, 128, 62, 0, 0, 128, 62, 0, 0, 128, 62, 0, 0, 128, 62, 0, 0, 128, 62, 0, 0, 128, 62 ]"){}
};

struct PreluBroadcastAlphaFixture : PreluFixture
{
    PreluBroadcastAlphaFixture() : PreluFixture(
        "[ 1, 1, 2, 3 ]",
        "[ 1, 3 ]",
        "[ 1, 1, 2, 3 ]",
        "[ 0 ]",
        "\"data\": [ 0, 0, 128, 62, 0, 0, 128, 62, 0, 0, 128, 62 ]"){}
};

struct PreluDynamicTensorFixture : PreluFixture
{
    PreluDynamicTensorFixture() : PreluFixture("[ 2, 3 ]",
                                               "[ 1, 1 ]",
                                               "[]",
                                               "[ 0 ]",
                                               "\"data\": [ 0, 0, 128, 62 ]") {}
};

TEST_CASE_FIXTURE(SimplePreluFixture, "SimplePrelu")
{
  RunTest<2, armnn::DataType::Float32>(
      0,
      {{"input0", { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f }},{"input1", { 0.25f }}},
      {{"output", { -3.5f, 2.f, 0.f, 1.f, -1.25f, 14.f }}});
}

TEST_CASE_FIXTURE(PreluConstAlphaFixture, "PreluConstAlpha")
{
  RunTest<3, armnn::DataType::Float32>(
      0,
      {{"input0", { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f }}},
      {{"output", { -3.5f, 2.f, 0.f, 1.f, -1.25f, 14.f }}});
}

TEST_CASE_FIXTURE(PreluBroadcastAlphaFixture, "PreluBroadcastAlpha")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"input0", { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f }}},
      {{"output", { -3.5f, 2.f, 0.f, 1.f, -1.25f, 14.f }}});
}

TEST_CASE_FIXTURE(PreluDynamicTensorFixture, "PreluDynamicTensor")
{
  RunTest<2, armnn::DataType::Float32, armnn::DataType::Float32>(
      0,
      {{"input0", { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f }}},
      {{"output", { -3.5f, 2.f, 0.f, 1.f, -1.25f, 14.f }}},
      true);
}

TEST_CASE_FIXTURE(PreluNetworkFixture, "PreluNetwork")
{
  RunTest<3, armnn::DataType::Float32>(
      0,
      {{"input0", { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f }}},
      {{"output", { -21.f, 12.f, 0.f, 6.f, -7.5f, 84.f }}});
}

}
