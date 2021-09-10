//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_Slice")
{
struct SliceFixture : public ParserFlatbuffersFixture
{
    explicit SliceFixture(const std::string & inputShape,
                          const std::string & outputShape,
                          const std::string & beginData,
                          const std::string & sizeData)
    {
        m_JsonString = R"(
            {
                  "version": 3,
                  "operator_codes": [
                    {
                      "builtin_code": "SLICE",
                      "version": 1
                    }
                  ],
                  "subgraphs": [
                    {
                      "tensors": [
                        {
                          "shape": )" + inputShape + R"(,
                          "type": "FLOAT32",
                          "buffer": 0,
                          "name": "inputTensor",
                          "quantization": {
                            "min": [
                              0.0
                            ],
                            "max": [
                              255.0
                            ],
                            "details_type": 0,
                            "quantized_dimension": 0
                          },
                          "is_variable": false
                        },
                        {
                          "shape": )" + outputShape + R"(,
                          "type": "FLOAT32",
                          "buffer": 1,
                          "name": "outputTensor",
                          "quantization": {
                            "details_type": 0,
                            "quantized_dimension": 0
                          },
                          "is_variable": false
                        })";
        m_JsonString += R"(,
                            {
                            "shape": [
                                3
                            ],
                            "type": "INT32",
                            "buffer": 2,
                            "name": "beginTensor",
                            "quantization": {
                            }
                            })";
        m_JsonString += R"(,
                            {
                            "shape": [
                                3
                            ],
                            "type": "INT32",
                            "buffer": 3,
                            "name": "sizeTensor",
                            "quantization": {
                            }
                            })";
        m_JsonString += R"(],
                      "inputs": [
                        0
                      ],
                      "outputs": [
                        1
                      ],
                      "operators": [
                        {
                          "opcode_index": 0,
                          "inputs": [
                            0,
                            2,
                            3)";
        m_JsonString += R"(],
                          "outputs": [
                            1
                          ],
                          mutating_variable_inputs: [
                          ]
                        }
                      ]
                    }
                  ],
                  "description": "TOCO Converted.",
                  "buffers": [
                    { },
                    { })";
        m_JsonString += R"(,{"data": )" + beginData + R"( })";
        m_JsonString += R"(,{"data": )" + sizeData + R"( })";
        m_JsonString += R"(
                  ]
                }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SliceFixtureSingleDim : SliceFixture
{
    SliceFixtureSingleDim() : SliceFixture("[ 3, 2, 3 ]",
                                           "[ 1, 1, 3 ]",
                                           "[ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]",
                                           "[ 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(SliceFixtureSingleDim, "SliceSingleDim")
{
    RunTest<3, armnn::DataType::Float32>(
      0,
      {{"inputTensor", { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 }}},
      {{"outputTensor", { 3, 3, 3 }}});

    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({1,1,3})));
}

struct SliceFixtureD123 : SliceFixture
{
    SliceFixtureD123() : SliceFixture("[ 3, 2, 3 ]",
                                      "[ 1, 2, 3 ]",
                                      "[ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]",
                                      "[ 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(SliceFixtureD123, "SliceD123")
{
    RunTest<3, armnn::DataType::Float32>(
        0,
        {{"inputTensor", { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 }}},
        {{"outputTensor", { 3, 3, 3, 4, 4, 4 }}});

    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({1,2,3})));
}

struct SliceFixtureD213 : SliceFixture
{
    SliceFixtureD213() : SliceFixture("[ 3, 2, 3 ]",
                                      "[ 2, 1, 3 ]",
                                      "[ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]",
                                      "[ 2, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0 ]") {}
};

TEST_CASE_FIXTURE(SliceFixtureD213, "SliceD213")
{
    RunTest<3, armnn::DataType::Float32>(
        0,
        {{"inputTensor", { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 }}},
        {{"outputTensor", { 3, 3, 3, 5, 5, 5 }}});

    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({2,1,3})));
}

struct DynamicSliceFixtureD213 : SliceFixture
{
    DynamicSliceFixtureD213() : SliceFixture("[ 3, 2, 3 ]",
                                            "[ ]",
                                              "[ 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 ]",
                                                "[ 255, 255, 255, 255, 1, 0, 0, 0, 255, 255, 255, 255 ]") {}
};

TEST_CASE_FIXTURE(DynamicSliceFixtureD213, "DynamicSliceD213")
{
    RunTest<3, armnn::DataType::Float32, armnn::DataType::Float32>(
        0,
        {{"inputTensor", { 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6 }}},
        {{"outputTensor", { 3, 3, 3, 5, 5, 5 }}},
        true);
}
}