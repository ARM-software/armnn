//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_BatchToSpaceND")
{
struct BatchToSpaceNDFixture : public ParserFlatbuffersFixture
{
    explicit BatchToSpaceNDFixture(const std::string & inputShape,
                                   const std::string & outputShape,
                                   const std::string & blockShapeData,
                                   const std::string & cropsData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "BATCH_TO_SPACE_ND" } ],
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
                             "name": "cropsTensor",
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
                    { "data": )" + cropsData + R"(, },
                ]
            }
        )";
      Setup();
    }
};

struct BatchToSpaceNDFixtureTest1 : public BatchToSpaceNDFixture
{
    BatchToSpaceNDFixtureTest1() : BatchToSpaceNDFixture("[ 4, 2, 2, 1 ]",
                                                         "[ 1, 4, 4, 1 ]",
                                                         "[ 2,0,0,0, 2,0,0,0 ]",
                                                         "[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]") {}
};

TEST_CASE_FIXTURE(BatchToSpaceNDFixtureTest1, "BatchToSpaceNDTest1")
{
    RunTest<4, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { // Batch 0, Height 0, Width (2) x Channel (1)
                              1.0f, 3.0f,
                              // Batch 0, Height 1, Width (2) x Channel (1)
                              9.0f, 11.0f,

                              // Batch 1, Height 0, Width (2) x Channel (1)
                              2.0f, 4.0f,
                              // Batch 1, Height 1, Width (2) x Channel (1)
                              10.0f, 12.0f,

                              // Batch 2, Height 0, Width (2) x Channel (1)
                              5.0f, 7.0f,
                              // Batch 2, Height 1, Width (2) x Channel (1)
                              13.0f, 15.0f,

                              // Batch 3, Height 0, Width (2) x Channel (3)
                              6.0f, 8.0f,
                              // Batch 3, Height 1, Width (2) x Channel (1)
                              14.0f, 16.0f }}},
         {{ "outputTensor", { 1.0f,   2.0f,  3.0f,  4.0f,
                              5.0f,   6.0f,  7.0f,  8.0f,
                              9.0f,  10.0f, 11.0f,  12.0f,
                              13.0f, 14.0f, 15.0f,  16.0f }}});
}

struct BatchToSpaceNDFixtureTest2 : public BatchToSpaceNDFixture
{
    BatchToSpaceNDFixtureTest2() : BatchToSpaceNDFixture("[ 4, 1, 1, 1 ]",
                                                         "[ 1, 2, 2, 1 ]",
                                                         "[ 2,0,0,0, 2,0,0,0 ]",
                                                         "[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]") {}
};

TEST_CASE_FIXTURE(BatchToSpaceNDFixtureTest2, "ParseBatchToSpaceNDTest2")
{
    RunTest<4, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1.0f, 2.0f, 3.0f, 4.0f }}},
         {{ "outputTensor", { // Batch 0, Height 0, Width (2) x Channel (1)
                              1.0f, 2.0f, 3.0f, 4.0f }}});
}

struct BatchToSpaceNDFixtureTest3 : public BatchToSpaceNDFixture
{
    BatchToSpaceNDFixtureTest3() : BatchToSpaceNDFixture("[ 4, 1, 1, 3 ]",
                                                         "[ 1, 2, 2, 3 ]",
                                                         "[ 2,0,0,0, 2,0,0,0 ]",
                                                         "[ 0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0 ]") {}
};

TEST_CASE_FIXTURE(BatchToSpaceNDFixtureTest3, "ParseBatchToSpaceNDTest3")
{
    RunTest<4, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }}},
         {{ "outputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f }}});
}

}
