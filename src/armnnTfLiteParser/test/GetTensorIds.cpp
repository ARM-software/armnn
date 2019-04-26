//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

using armnnTfLiteParser::TfLiteParser;
using ModelPtr = TfLiteParser::ModelPtr;

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

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

BOOST_FIXTURE_TEST_CASE(GetEmptyInputTensorIds, GetEmptyTensorIdsFixture)
{
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    std::vector<int32_t> expectedIds = { };
    std::vector<int32_t> inputTensorIds = TfLiteParser::GetInputTensorIds(model, 0, 0);
    BOOST_CHECK_EQUAL_COLLECTIONS(expectedIds.begin(), expectedIds.end(),
                                  inputTensorIds.begin(), inputTensorIds.end());
}

BOOST_FIXTURE_TEST_CASE(GetEmptyOutputTensorIds, GetEmptyTensorIdsFixture)
{
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    std::vector<int32_t> expectedIds = { };
    std::vector<int32_t> outputTensorIds = TfLiteParser::GetOutputTensorIds(model, 0, 0);
    BOOST_CHECK_EQUAL_COLLECTIONS(expectedIds.begin(), expectedIds.end(),
                                  outputTensorIds.begin(), outputTensorIds.end());
}

BOOST_FIXTURE_TEST_CASE(GetInputTensorIds, GetInputOutputTensorIdsFixture)
{
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    std::vector<int32_t> expectedInputIds = { 0, 1, 2 };
    std::vector<int32_t> inputTensorIds = TfLiteParser::GetInputTensorIds(model, 0, 0);
    BOOST_CHECK_EQUAL_COLLECTIONS(expectedInputIds.begin(), expectedInputIds.end(),
                                  inputTensorIds.begin(), inputTensorIds.end());
}

BOOST_FIXTURE_TEST_CASE(GetOutputTensorIds, GetInputOutputTensorIdsFixture)
{
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    std::vector<int32_t> expectedOutputIds = { 3 };
    std::vector<int32_t> outputTensorIds = TfLiteParser::GetOutputTensorIds(model, 0, 0);
    BOOST_CHECK_EQUAL_COLLECTIONS(expectedOutputIds.begin(), expectedOutputIds.end(),
                                  outputTensorIds.begin(), outputTensorIds.end());
}

BOOST_FIXTURE_TEST_CASE(GetInputTensorIdsNullModel, GetInputOutputTensorIdsFixture)
{
    BOOST_CHECK_THROW(TfLiteParser::GetInputTensorIds(nullptr, 0, 0), armnn::ParseException);
}

BOOST_FIXTURE_TEST_CASE(GetOutputTensorIdsNullModel, GetInputOutputTensorIdsFixture)
{
    BOOST_CHECK_THROW(TfLiteParser::GetOutputTensorIds(nullptr, 0, 0), armnn::ParseException);
}

BOOST_FIXTURE_TEST_CASE(GetInputTensorIdsInvalidSubgraph, GetInputOutputTensorIdsFixture)
{
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    BOOST_CHECK_THROW(TfLiteParser::GetInputTensorIds(model, 1, 0), armnn::ParseException);
}

BOOST_FIXTURE_TEST_CASE(GetOutputTensorIdsInvalidSubgraph, GetInputOutputTensorIdsFixture)
{
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    BOOST_CHECK_THROW(TfLiteParser::GetOutputTensorIds(model, 1, 0), armnn::ParseException);
}

BOOST_FIXTURE_TEST_CASE(GetInputTensorIdsInvalidOperator, GetInputOutputTensorIdsFixture)
{
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    BOOST_CHECK_THROW(TfLiteParser::GetInputTensorIds(model, 0, 1), armnn::ParseException);
}

BOOST_FIXTURE_TEST_CASE(GetOutputTensorIdsInvalidOperator, GetInputOutputTensorIdsFixture)
{
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    BOOST_CHECK_THROW(TfLiteParser::GetOutputTensorIds(model, 0, 1), armnn::ParseException);
}

BOOST_AUTO_TEST_SUITE_END()
