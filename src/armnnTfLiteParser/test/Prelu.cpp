//
// Copyright © 2021 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <string>

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

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

struct SimplePreluFixture : PreluFixture
{
    SimplePreluFixture() : PreluFixture("[ 2, 3 ]",
                                        "[ 1, 1 ]",
                                        "[ 2, 3 ]",
                                        "[ 0, 1 ]",
                                        "") {}
};

struct PreluConstAlphaFixture : PreluFixture
{
    PreluConstAlphaFixture() : PreluFixture(
        "[ 2, 3 ]",
        "[ 2, 3 ]",
        "[ 2, 3 ]",
        "[ 0 ]",
        "\"data\": [ 0, 0, 128, 62, 0, 0, 128, 62, 0, 0, 128, 62, 0, 0, 128, 62, 0, 0, 128, 62, 0, 0, 128, 62 ]"){}
};

struct PreluDynamicTensorFixture : PreluFixture
{
    PreluDynamicTensorFixture() : PreluFixture("[ 2, 3 ]",
                                               "[ 1, 1 ]",
                                               "[]",
                                               "[ 0 ]",
                                               "\"data\": [ 0, 0, 128, 62 ]") {}
};

BOOST_FIXTURE_TEST_CASE(SimplePrelu, SimplePreluFixture)
{
  RunTest<2, armnn::DataType::Float32>(
      0,
      {{"input0", { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f }},{"input1", { 0.25f }}},
      {{"output", { -3.5f, 2.f, 0.f, 1.f, -1.25f, 14.f }}});
}

BOOST_FIXTURE_TEST_CASE(PreluConstAlpha, PreluConstAlphaFixture)
{
  RunTest<2, armnn::DataType::Float32>(
      0,
      {{"input0", { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f }}},
      {{"output", { -3.5f, 2.f, 0.f, 1.f, -1.25f, 14.f }}});
}

BOOST_FIXTURE_TEST_CASE(PreluDynamicTensor, PreluDynamicTensorFixture)
{
  RunTest<2, armnn::DataType::Float32, armnn::DataType::Float32>(
      0,
      {{"input0", { -14.f, 2.f, 0.f, 1.f, -5.f, 14.f }}},
      {{"output", { -3.5f, 2.f, 0.f, 1.f, -1.25f, 14.f }}},
      true);
}

BOOST_AUTO_TEST_SUITE_END()
