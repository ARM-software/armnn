//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Maximum")
{
struct MaximumFixture : public ParserFlatbuffersFixture
{
    explicit MaximumFixture(const std::string & inputShape1,
                            const std::string & inputShape2,
                            const std::string & outputShape)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "MAXIMUM" } ],
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


struct MaximumFixture4D4D : MaximumFixture
{
    MaximumFixture4D4D() : MaximumFixture("[ 1, 2, 2, 3 ]",
                                          "[ 1, 2, 2, 3 ]",
                                          "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MaximumFixture4D4D, "ParseMaximum4D4D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor1", { 0.0f, 1.0f, 2.0f,
                          3.0f, 4.0f, 5.0f,
                          6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f }},
      {"inputTensor2",  { 5.0f, 1.0f, 3.0f,
                          4.0f, 5.5f, 1.0f,
                          2.0f, 17.0f, 18.0f,
                          19.0f, 1.0f, 3.0f }}},
      {{"outputTensor", { 5.0f,  1.0f, 3.0f,
                          4.0f,  5.5f, 5.0f,
                          6.0f,  17.0f, 18.0f,
                          19.0f, 10.0f, 11.0f }}});
}

struct MaximumBroadcastFixture4D4D : MaximumFixture
{
    MaximumBroadcastFixture4D4D() : MaximumFixture("[ 1, 1, 2, 1 ]",
                                                   "[ 1, 2, 1, 3 ]",
                                                   "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MaximumBroadcastFixture4D4D, "ParseMaximumBroadcast4D4D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor1", { 2.0f, 4.0f }},
      {"inputTensor2",  { 1.0f, 2.0f, 3.0f,
                          4.0f, 5.0f, 6.0f }}},
      {{"outputTensor", { 2.0f, 2.0f, 3.0f,
                          4.0f, 4.0f, 4.0f,
                          4.0f, 5.0f, 6.0f,
                          4.0f, 5.0f, 6.0f }}});
}

struct MaximumBroadcastFixture4D1D : MaximumFixture
{
    MaximumBroadcastFixture4D1D() : MaximumFixture("[ 1, 2, 2, 3 ]",
                                                   "[ 1 ]",
                                                   "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MaximumBroadcastFixture4D1D, "ParseMaximumBroadcast4D1D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor1", { 0.0f, 1.0f, 2.0f,
                          3.0f, 4.0f, 5.0f,
                          6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f }},
      {"inputTensor2",  { 5.0f }}},
      {{"outputTensor", { 5.0f, 5.0f, 5.0f,
                          5.0f, 5.0f, 5.0f,
                          6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f }}});
}

struct MaximumBroadcastFixture1D4D : MaximumFixture
{
    MaximumBroadcastFixture1D4D() : MaximumFixture("[ 1 ]",
                                                   "[ 1, 2, 2, 3 ]",
                                                   "[ 1, 2, 2, 3 ]") {}
};

TEST_CASE_FIXTURE(MaximumBroadcastFixture1D4D, "ParseMaximumBroadcast1D4D")
{
  RunTest<4, armnn::DataType::Float32>(
      0,
      {{"inputTensor1", { 3.0f }},
      {"inputTensor2",  { 0.0f, 1.0f, 2.0f,
                          3.0f, 4.0f, 5.0f,
                          6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f }}},
      {{"outputTensor", { 3.0f, 3.0f, 3.0f,
                          3.0f, 4.0f, 5.0f,
                          6.0f, 7.0f, 8.0f,
                          9.0f, 10.0f, 11.0f }}});
}

}
