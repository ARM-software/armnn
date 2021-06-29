//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_SpaceToBatchND")
{
struct SpaceToBatchNDFixture : public ParserFlatbuffersFixture
{
    explicit SpaceToBatchNDFixture(const std::string & inputShape,
                                   const std::string & outputShape,
                                   const std::string & blockShapeData,
                                   const std::string & padListData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "SPACE_TO_BATCH_ND" } ],
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
                             "shape": )" + outputShape + R"(,
                             "type": "FLOAT32",
                             "buffer": 1,
                             "name": "outputTensor",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                             "shape": [ 2 ],
                             "type": "INT32",
                             "buffer": 2,
                             "name": "blockShapeTensor",
                             "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                             }
                        },
                        {
                             "shape": [ 2, 2 ],
                             "type": "INT32",
                             "buffer": 3,
                             "name": "padListTensor",
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
                            "inputs": [ 0, 2, 3 ],
                            "outputs": [ 1 ],
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + blockShapeData + R"(, },
                    { "data": )" + padListData + R"(, },
                ]
            }
        )";
      Setup();
    }
};

struct SpaceToBatchNDFixtureSimpleTest : public SpaceToBatchNDFixture
{
    SpaceToBatchNDFixtureSimpleTest() : SpaceToBatchNDFixture("[ 1, 4, 4, 1 ]",
                                                              "[ 4, 2, 2, 1 ]",
                                                              "[ 2,0,0,0, 2,0,0,0 ]",
                                                              "[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]") {}
};

TEST_CASE_FIXTURE(SpaceToBatchNDFixtureSimpleTest, "SpaceToBatchNdSimpleTest")
{
    RunTest<4, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1.0f,  2.0f,  3.0f,  4.0f,
                              5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f,
                             13.0f, 14.0f, 15.0f, 16.0f }}},
         {{ "outputTensor", { 1.0f, 3.0f,  9.0f, 11.0f,
                              2.0f, 4.0f, 10.0f, 12.0f,
                              5.0f, 7.0f, 13.0f, 15.0f,
                              6.0f, 8.0f, 14.0f, 16.0f }}});
}


struct SpaceToBatchNDFixtureMultipleInputBatchesTest : public SpaceToBatchNDFixture
{
    SpaceToBatchNDFixtureMultipleInputBatchesTest() : SpaceToBatchNDFixture("[ 2, 2, 4, 1 ]",
                                                                            "[ 8, 1, 2, 1 ]",
                                                                            "[ 2,0,0,0, 2,0,0,0 ]",
                                                                            "[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]") {}
};

TEST_CASE_FIXTURE(SpaceToBatchNDFixtureMultipleInputBatchesTest, "SpaceToBatchNdMultipleInputBatchesTest")
{
    RunTest<4, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1.0f,  2.0f,  3.0f,  4.0f,
                              5.0f,  6.0f,  7.0f,  8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f,
                             13.0f, 14.0f, 15.0f, 16.0f }}},
         {{ "outputTensor", {  1.0f, 3.0f,  9.0f, 11.0f,
                               2.0f, 4.0f, 10.0f, 12.0f,
                               5.0f, 7.0f, 13.0f, 15.0f,
                               6.0f, 8.0f, 14.0f, 16.0f }}});
}

struct SpaceToBatchNDFixturePaddingTest : public SpaceToBatchNDFixture
{
    SpaceToBatchNDFixturePaddingTest() : SpaceToBatchNDFixture("[ 1, 5, 2, 1 ]",
                                                               "[ 6, 2, 2, 1 ]",
                                                               "[ 3,0,0,0, 2,0,0,0 ]",
                                                               "[ 1,0,0,0, 0,0,0,0, 2,0,0,0, 0,0,0,0 ]") {}
};

TEST_CASE_FIXTURE(SpaceToBatchNDFixturePaddingTest, "SpaceToBatchNdPaddingTest")
{
    RunTest<4, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  {  1.0f,  2.0f,  3.0f,  4.0f, 5.0f,
                               6.0f,  7.0f,  8.0f,  9.0f, 10.0f }}},
         {{ "outputTensor", {  0.0f, 0.0f,
                               0.0f, 5.0f,

                               0.0f, 0.0f,
                               0.0f, 6.0f,

                               0.0f, 1.0f,
                               0.0f, 7.0f,

                               0.0f, 2.0f,
                               0.0f, 8.0f,

                               0.0f, 3.0f,
                               0.0f, 9.0f,

                               0.0f, 4.0f,
                               0.0f, 10.0f, }}});
}

}
