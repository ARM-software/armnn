//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_AvgPool2D")
{
struct AvgPool2DFixture : public ParserFlatbuffersFixture
{
    explicit AvgPool2DFixture(std::string inputdim, std::string outputdim, std::string dataType)
    {
        m_JsonString = R"(
        {
            "version": 3,
            "operator_codes": [ { "builtin_code": "AVERAGE_POOL_2D" } ],
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
            "description": "AvgPool2D test.",
            "buffers" : [ {}, {} ]
        })";

        SetupSingleInputSingleOutput("InputTensor", "OutputTensor");
    }
};


struct AvgPoolLiteFixtureUint1DOutput : AvgPool2DFixture
{
    AvgPoolLiteFixtureUint1DOutput() : AvgPool2DFixture("[ 1, 2, 2, 1 ]", "[ 1, 1, 1, 1 ]", "UINT8") {}
};

struct AvgPoolLiteFixtureFloat1DOutput : AvgPool2DFixture
{
    AvgPoolLiteFixtureFloat1DOutput() : AvgPool2DFixture("[ 1, 2, 2, 1 ]", "[ 1, 1, 1, 1 ]", "FLOAT32") {}
};

struct AvgPoolLiteFixture2DOutput : AvgPool2DFixture
{
    AvgPoolLiteFixture2DOutput() : AvgPool2DFixture("[ 1, 4, 4, 1 ]", "[ 1, 2, 2, 1 ]", "UINT8") {}
};

TEST_CASE_FIXTURE(AvgPoolLiteFixtureUint1DOutput, "AvgPoolLite1DOutput")
{
    RunTest<4, armnn::DataType::QAsymmU8>(0, {2, 3, 5, 2 }, { 3 });
}

TEST_CASE_FIXTURE(AvgPoolLiteFixtureFloat1DOutput, "AvgPoolLiteFloat1DOutput")
{
    RunTest<4, armnn::DataType::Float32>(0, { 2.0f, 3.0f, 5.0f, 2.0f },  { 3.0f });
}

TEST_CASE_FIXTURE(AvgPoolLiteFixture2DOutput, "AvgPoolLite2DOutput")
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0, { 1, 2, 2, 3, 5, 6, 7, 8, 3, 2, 1, 0, 1, 2, 3, 4 }, { 4, 5, 2, 2 });
}

TEST_CASE_FIXTURE(AvgPoolLiteFixtureFloat1DOutput, "IncorrectDataTypeError")
{
    CHECK_THROWS_AS((RunTest<4, armnn::DataType::QAsymmU8>(0, {2, 3, 5, 2 }, { 3 })), armnn::Exception);
}

}
