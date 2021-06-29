//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_MaxPool2D")
{
struct MaxPool2DFixture : public ParserFlatbuffersFixture
{
    explicit MaxPool2DFixture(std::string inputdim, std::string outputdim, std::string dataType)
    {
        m_JsonString = R"(
        {
            "version": 3,
            "operator_codes": [ { "builtin_code": "MAX_POOL_2D" } ],
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
                        "builtin_options_type": "Pool2DOptions",
                        "builtin_options":
                        {
                            "padding": "VALID",
                            "stride_w": 2,
                            "stride_h": 2,
                            "filter_width": 2,
                            "filter_height": 2,
                            "fused_activation_function": "NONE"
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


struct MaxPoolLiteFixtureUint1DOutput : MaxPool2DFixture
{
    MaxPoolLiteFixtureUint1DOutput() : MaxPool2DFixture("[ 1, 2, 2, 1 ]", "[ 1, 1, 1, 1 ]", "UINT8") {}
};

struct MaxPoolLiteFixtureFloat1DOutput : MaxPool2DFixture
{
    MaxPoolLiteFixtureFloat1DOutput() : MaxPool2DFixture("[ 1, 2, 2, 1 ]", "[ 1, 1, 1, 1 ]", "FLOAT32") {}
};

struct MaxPoolLiteFixtureUint2DOutput : MaxPool2DFixture
{
    MaxPoolLiteFixtureUint2DOutput() : MaxPool2DFixture("[ 1, 4, 4, 1 ]", "[ 1, 2, 2, 1 ]", "UINT8") {}
};

TEST_CASE_FIXTURE(MaxPoolLiteFixtureUint1DOutput, "MaxPoolLiteUint1DOutput")
{
    RunTest<4, armnn::DataType::QAsymmU8>(0, { 2, 3, 5, 2 }, { 5 });
}

TEST_CASE_FIXTURE(MaxPoolLiteFixtureFloat1DOutput, "MaxPoolLiteFloat1DOutput")
{
    RunTest<4, armnn::DataType::Float32>(0, { 2.0f, 3.0f, 5.0f, 2.0f },  { 5.0f });
}

TEST_CASE_FIXTURE(MaxPoolLiteFixtureUint2DOutput, "MaxPoolLiteUint2DOutput")
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0, { 1, 2, 2, 3, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4 }, { 6, 8, 3, 4 });
}

TEST_CASE_FIXTURE(MaxPoolLiteFixtureFloat1DOutput, "MaxPoolIncorrectDataTypeError")
{
    CHECK_THROWS_AS((RunTest<4, armnn::DataType::QAsymmU8>(0, { 2, 3, 5, 2 }, { 5 })), armnn::Exception);
}

}
