//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_Transpose")
{
struct TransposeFixture : public ParserFlatbuffersFixture
{
    explicit TransposeFixture(const std::string & inputShape,
                              const std::string & permuteData,
                              const std::string & outputShape)
    {
        m_JsonString = R"(
            {
                  "version": 3,
                  "operator_codes": [
                    {
                      "builtin_code": "TRANSPOSE",
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
                            "name": "permuteTensor",
                            "quantization": {
                              "details_type": 0,
                              "quantized_dimension": 0
                            },
                            "is_variable": false
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
                            0)";
        m_JsonString += R"(,2)";
        m_JsonString += R"(],
                          "outputs": [
                            1
                          ],
                          "builtin_options_type": "TransposeOptions",
                          "builtin_options": {
                          },
                          "custom_options_format": "FLEXBUFFERS"
                        }
                      ]
                    }
                  ],
                  "description": "TOCO Converted.",
                  "buffers": [
                    { },
                    { })";
        if (!permuteData.empty())
        {
            m_JsonString += R"(,{"data": )" + permuteData + R"( })";
        }
        m_JsonString += R"(
                  ]
                }
        )";
        Setup();
    }
};

// Note that this assumes the Tensorflow permutation vector implementation as opposed to the armnn implemenation.
struct TransposeFixtureWithPermuteData : TransposeFixture
{
    TransposeFixtureWithPermuteData() : TransposeFixture("[ 2, 2, 3 ]",
                                                         "[ 0, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0 ]",
                                                         "[ 2, 3, 2 ]") {}
};

TEST_CASE_FIXTURE(TransposeFixtureWithPermuteData, "TransposeWithPermuteData")
{
    RunTest<3, armnn::DataType::Float32>(
      0,
      {{"inputTensor", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }}},
      {{"outputTensor", { 1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12 }}});

    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({2,3,2})));
}

// Tensorflow default permutation behavior assumes no permute argument will create permute vector [n-1...0],
// where n is the number of dimensions of the input tensor
// In this case we should get output shape 3,2,2 given default permutation vector 2,1,0
struct TransposeFixtureWithoutPermuteData : TransposeFixture
{
    TransposeFixtureWithoutPermuteData() : TransposeFixture("[ 2, 2, 3 ]",
                                                            "[ 2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ]",
                                                            "[ 3, 2, 2 ]") {}
};

TEST_CASE_FIXTURE(TransposeFixtureWithoutPermuteData, "TransposeWithoutPermuteDims")
{
    RunTest<3, armnn::DataType::Float32>(
        0,
        {{"inputTensor", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }}},
        {{"outputTensor", { 1, 7, 4, 10, 2, 8, 5, 11, 3, 9, 6, 12 }}});

    CHECK((m_Parser->GetNetworkOutputBindingInfo(0, "outputTensor").second.GetShape()
                == armnn::TensorShape({3,2,2})));
}

}