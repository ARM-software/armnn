//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_LRN")
{
struct LRNFixture : public ParserFlatbuffersFixture
{
    explicit LRNFixture(std::string inputdim, std::string outputdim, std::string dataType)
    {
        m_JsonString = R"(
        {
            "version": 3,
            "operator_codes": [ { "builtin_code": "LOCAL_RESPONSE_NORMALIZATION" } ],
            "subgraphs": [
            {
                "tensors": [
                {
                    "shape": )"
                       + outputdim
                       + R"(,
                    "type": )"
                       + dataType
                       + R"(,
                            "buffer": 0,
                            "name": "OutputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ]
                            }
                },
                {
                    "shape": )"
                       + inputdim
                       + R"(,
                    "type": )"
                       + dataType
                       + R"(,
                            "buffer": 1,
                            "name": "InputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ]
                            }
                }
                ],
                "inputs": [ 1 ],
                "outputs": [ 0 ],
                "operators": [ {
                        "opcode_index": 0,
                        "inputs": [ 1 ],
                        "outputs": [ 0 ],
                        "builtin_options_type": "LocalResponseNormalizationOptions",
                        "builtin_options":
                        {
                            "radius": 2,
                            "bias": 1.0,
                            "alpha": 1.0,
                            "beta": 0.5
                        },
                        "custom_options_format": "FLEXBUFFERS"
                    } ]
                }
            ],
            "description": "MaxPool2D test.",
            "buffers" : [ {}, {} ]
        })";

        SetupSingleInputSingleOutput("InputTensor", "OutputTensor");
    }
};

struct LRNLiteFixtureFloat4DOutput : LRNFixture
{
    LRNLiteFixtureFloat4DOutput() : LRNFixture("[ 1, 1, 4, 4 ]", "[ 1, 1, 4, 4 ]", "FLOAT32") {}
};

TEST_CASE_FIXTURE(LRNLiteFixtureFloat4DOutput, "LRNLiteFloat4DOutput")
{
    RunTest<4, armnn::DataType::Float32>(0,
                                         {
                                             2.0f, 3.0f, 5.0f, 2.0f,
                                             2.0f, 3.0f, 5.0f, 2.0f,
                                             2.0f, 3.0f, 5.0f, 2.0f,
                                             2.0f, 3.0f, 5.0f, 2.0f
                                         },
                                         {
                                             0.320256f, 0.457496f, 0.762493f, 0.320256f,
                                             0.320256f, 0.457496f, 0.762493f, 0.320256f,
                                             0.320256f, 0.457496f, 0.762493f, 0.320256f,
                                             0.320256f, 0.457496f, 0.762493f, 0.320256f
                                         });
}

TEST_CASE_FIXTURE(LRNLiteFixtureFloat4DOutput, "LRNIncorrectDataTypeError")
{
    CHECK_THROWS_AS((RunTest<4, armnn::DataType::QAsymmU8>(0, { 2, 3, 5, 2 }, { 5 })), armnn::Exception);
}

}
