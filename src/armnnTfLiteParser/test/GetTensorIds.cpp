//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

using armnnTfLiteParser::TfLiteParserImpl;
using ModelPtr = TfLiteParserImpl::ModelPtr;

TEST_SUITE("TensorflowLiteParser_GetTensorIds")
{
struct GetTensorIdsFixture : public ParserFlatbuffersFixture
{
    explicit GetTensorIdsFixture(const std::string& inputs, const std::string& outputs)
    {
        m_JsonString = R"(
        {
            "version": 3,
            "operator_codes": [ { "builtin_code": "AVERAGE_POOL_2D" } ],
            "subgraphs": [
            {
                "tensors": [
                {
                    "shape": [ 1, 1, 1, 1 ] ,
                    "type": "UINT8",
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
                    "shape": [ 1, 2, 2, 1 ] ,
                    "type": "UINT8",
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
                        "inputs": )"
                            + inputs
                            + R"(,
                        "outputs": )"
                            + outputs
                            + R"(,
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
            "description": "Test loading a model",
            "buffers" : [ {}, {} ]
        })";

        ReadStringToBinary();
    }
};

struct GetEmptyTensorIdsFixture : GetTensorIdsFixture
{
    GetEmptyTensorIdsFixture() : GetTensorIdsFixture("[ ]", "[ ]") {}
};

struct GetInputOutputTensorIdsFixture : GetTensorIdsFixture
{
    GetInputOutputTensorIdsFixture() : GetTensorIdsFixture("[ 0, 1, 2 ]", "[ 3 ]") {}
};

TEST_CASE_FIXTURE(GetEmptyTensorIdsFixture, "GetEmptyInputTensorIds")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    std::vector<int32_t> expectedIds = { };
    std::vector<int32_t> inputTensorIds = TfLiteParserImpl::GetInputTensorIds(model, 0, 0);
    CHECK(std::equal(expectedIds.begin(), expectedIds.end(),
                                  inputTensorIds.begin(), inputTensorIds.end()));
}

TEST_CASE_FIXTURE(GetEmptyTensorIdsFixture, "GetEmptyOutputTensorIds")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    std::vector<int32_t> expectedIds = { };
    std::vector<int32_t> outputTensorIds = TfLiteParserImpl::GetOutputTensorIds(model, 0, 0);
    CHECK(std::equal(expectedIds.begin(), expectedIds.end(),
                                  outputTensorIds.begin(), outputTensorIds.end()));
}

TEST_CASE_FIXTURE(GetInputOutputTensorIdsFixture, "GetInputTensorIds")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    std::vector<int32_t> expectedInputIds = { 0, 1, 2 };
    std::vector<int32_t> inputTensorIds = TfLiteParserImpl::GetInputTensorIds(model, 0, 0);
    CHECK(std::equal(expectedInputIds.begin(), expectedInputIds.end(),
                                  inputTensorIds.begin(), inputTensorIds.end()));
}

TEST_CASE_FIXTURE(GetInputOutputTensorIdsFixture, "GetOutputTensorIds")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    std::vector<int32_t> expectedOutputIds = { 3 };
    std::vector<int32_t> outputTensorIds = TfLiteParserImpl::GetOutputTensorIds(model, 0, 0);
    CHECK(std::equal(expectedOutputIds.begin(), expectedOutputIds.end(),
                                  outputTensorIds.begin(), outputTensorIds.end()));
}

TEST_CASE_FIXTURE(GetInputOutputTensorIdsFixture, "GetInputTensorIdsNullModel")
{
    CHECK_THROWS_AS(TfLiteParserImpl::GetInputTensorIds(nullptr, 0, 0), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetInputOutputTensorIdsFixture, "GetOutputTensorIdsNullModel")
{
    CHECK_THROWS_AS(TfLiteParserImpl::GetOutputTensorIds(nullptr, 0, 0), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetInputOutputTensorIdsFixture, "GetInputTensorIdsInvalidSubgraph")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetInputTensorIds(model, 1, 0), armnn::ParseException);
}

TEST_CASE_FIXTURE( GetInputOutputTensorIdsFixture, "GetOutputTensorIdsInvalidSubgraph")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetOutputTensorIds(model, 1, 0), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetInputOutputTensorIdsFixture, "GetInputTensorIdsInvalidOperator")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetInputTensorIds(model, 0, 1), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetInputOutputTensorIdsFixture, "GetOutputTensorIdsInvalidOperator")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetOutputTensorIds(model, 0, 1), armnn::ParseException);
}

}
