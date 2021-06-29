//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Minimum")
{
struct MinimumFixture : public ParserFlatbuffersFixture
{
    explicit MinimumFixture(const std::string & inputShape1,
                            const std::string & inputShape2,
                            const std::string & outputShape)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "MINIMUM" } ],
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


struct MinimumFixture4D : MinimumFixture
{
    MinimumFixture4D() : MinimumFixture("[ 1, 2, 2, 3 ]",
                                        "[ 1, 2, 2, 3 ]",
                                        "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MinimumFixture4D, "ParseMinimum4D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor1", { 0.0f,  1.0f,  2.0f,
                          3.0f,  4.0f,  5.0f,
                          6.0f,  7.0f,  8.0f,
                          9.0f, 10.0f, 11.0f }},
      {"inputTensor2",  { 0.0f, 0.0f, 0.0f,
                          5.0f, 5.0f, 5.0f,
                          7.0f, 7.0f, 7.0f,
                          9.0f, 9.0f, 9.0f }}},
      {{"outputTensor", { 0.0f, 0.0f, 0.0f,
                          3.0f, 4.0f, 5.0f,
                          6.0f, 7.0f, 7.0f,
                          9.0f, 9.0f, 9.0f }}});
}

struct MinimumBroadcastFixture4D : MinimumFixture
{
    MinimumBroadcastFixture4D() : MinimumFixture("[ 1, 1, 2, 1 ]",
                                                 "[ 1, 2, 1, 3 ]",
                                                 "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MinimumBroadcastFixture4D, "ParseMinimumBroadcast4D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor1", { 2.0f,
                          4.0f }},
      {"inputTensor2",  { 1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f }}},
      {{"outputTensor", { 1.0f, 2.0f, 2.0f,
                          1.0f, 2.0f, 3.0f,
                          2.0f, 2.0f, 2.0f,
                          4.0f, 4.0f, 4.0f }}});
}

struct MinimumBroadcastFixture4D1D : MinimumFixture
{
    MinimumBroadcastFixture4D1D() : MinimumFixture("[ 1, 2, 2, 3 ]",
                                                   "[ 1 ]",
                                                   "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MinimumBroadcastFixture4D1D, "ParseMinimumBroadcast4D1D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor1", { 0.0f,  1.0f,  2.0f,
                          3.0f,  4.0f,  5.0f,
                          6.0f,  7.0f,  8.0f,
                          9.0f, 10.0f, 11.0f }},
      {"inputTensor2",  { 5.0f }}},
      {{"outputTensor", {  0.0f, 1.0f, 2.0f,
                           3.0f, 4.0f, 5.0f,
                           5.0f, 5.0f, 5.0f,
                           5.0f, 5.0f, 5.0f }}});
}

struct MinimumBroadcastFixture1D4D : MinimumFixture
{
    MinimumBroadcastFixture1D4D() : MinimumFixture("[ 3 ]",
                                                   "[ 1, 2, 2, 3 ]",
                                                   "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MinimumBroadcastFixture1D4D, "ParseMinimumBroadcast1D4D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor1", { 5.0f,  6.0f,  7.0f }},
      {"inputTensor2",  { 0.0f,  1.0f,  2.0f,
                          3.0f,  4.0f,  5.0f,
                          6.0f,  7.0f,  8.0f,
                          9.0f, 10.0f, 11.0f }}},
      {{"outputTensor", { 0.0f, 1.0f, 2.0f,
                          3.0f, 4.0f, 5.0f,
                          5.0f, 6.0f, 7.0f,
                          5.0f, 6.0f, 7.0f }}});
}

struct MinimumBroadcastFixture2D0D : public ParserFlatbuffersFixture
{
    explicit MinimumBroadcastFixture2D0D()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "MINIMUM" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 1, 2 ],
                            "type": "FLOAT32",
                            "buffer": 0,
                            "name": "input0",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": [ ],
                            "type": "FLOAT32",
                            "buffer": 2,
                            "name": "input1",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": [ 1, 2 ] ,
                            "type": "FLOAT32",
                            "buffer": 1,
                            "name": "output",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 2 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1 ],
                            "outputs": [ 2 ],
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": [ 0, 0, 0, 64 ] }
                ]
            }
        )";
        Setup();
    }
};

TEST_CASE_FIXTURE(MinimumBroadcastFixture2D0D, "ParseMinimumBroadcast2D0D")
{
    RunTest<2, armnn::DataType::Float32>(
            0,
            {{"input0", { 1.0f, 5.0f }}},
            {{"output", { 1.0f, 2.0f }}});
}

}
