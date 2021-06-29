//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

using armnnTfLiteParser::TfLiteParserImpl;
using ModelPtr = TfLiteParserImpl::ModelPtr;
using TensorRawPtr = TfLiteParserImpl::TensorRawPtr;

TEST_SUITE("TensorflowLiteParser_GetSubgraphInputsOutputs")
{
struct GetSubgraphInputsOutputsMainFixture : public ParserFlatbuffersFixture
{
    explicit GetSubgraphInputsOutputsMainFixture(const std::string& inputs, const std::string& outputs)
    {
        m_JsonString = R"(
        {
            "version": 3,
            "operator_codes": [ { "builtin_code": "AVERAGE_POOL_2D" }, { "builtin_code": "CONV_2D" } ],
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
                                "min": [ -1.2 ],
                                "max": [ 25.5 ],
                                "scale": [ 0.25 ],
                                "zero_point": [ 10 ]
                            }
                }
                ],
                "inputs": )"
                            + inputs
                            + R"(,
                "outputs": )"
                            + outputs
                            + R"(,
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
                },
                {
                    "tensors": [
                        {
                            "shape": [ 1, 3, 3, 1 ],
                            "type": "UINT8",
                            "buffer": 0,
                            "name": "ConvInputTensor",
                            "quantization": {
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": [ 1, 1, 1, 1 ],
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "ConvOutputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 511.0 ],
                                "scale": [ 2.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": [ 1, 3, 3, 1 ],
                            "type": "UINT8",
                            "buffer": 2,
                            "name": "filterTensor",
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
                            "inputs": [ 0, 2 ],
                            "outputs": [ 1 ],
                            "builtin_options_type": "Conv2DOptions",
                            "builtin_options": {
                                "padding": "VALID",
                                "stride_w": 1,
                                "stride_h": 1,
                                "fused_activation_function": "NONE"
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                }
            ],
            "description": "Test Subgraph Inputs Outputs",
            "buffers" : [
                    { },
                    { },
                    { "data": [ 2,1,0, 6,2,1, 4,1,2 ], },
                    { },
                ]
        })";

        ReadStringToBinary();
    }

};

struct GetEmptySubgraphInputsOutputsFixture : GetSubgraphInputsOutputsMainFixture
{
    GetEmptySubgraphInputsOutputsFixture() : GetSubgraphInputsOutputsMainFixture("[ ]", "[ ]") {}
};

struct GetSubgraphInputsOutputsFixture : GetSubgraphInputsOutputsMainFixture
{
    GetSubgraphInputsOutputsFixture() : GetSubgraphInputsOutputsMainFixture("[ 1 ]", "[ 0 ]") {}
};

TEST_CASE_FIXTURE(GetEmptySubgraphInputsOutputsFixture, "GetEmptySubgraphInputs")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorIdRawPtrVector subgraphTensors = TfLiteParserImpl::GetSubgraphInputs(model, 0);
    CHECK_EQ(0, subgraphTensors.size());
}

TEST_CASE_FIXTURE(GetEmptySubgraphInputsOutputsFixture, "GetEmptySubgraphOutputs")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorIdRawPtrVector subgraphTensors = TfLiteParserImpl::GetSubgraphOutputs(model, 0);
    CHECK_EQ(0, subgraphTensors.size());
}

TEST_CASE_FIXTURE(GetSubgraphInputsOutputsFixture, "GetSubgraphInputs")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorIdRawPtrVector subgraphTensors = TfLiteParserImpl::GetSubgraphInputs(model, 0);
    CHECK_EQ(1, subgraphTensors.size());
    CHECK_EQ(1, subgraphTensors[0].first);
    CheckTensors(subgraphTensors[0].second, 4, { 1, 2, 2, 1 }, tflite::TensorType::TensorType_UINT8, 1,
                      "InputTensor", { -1.2f }, { 25.5f }, { 0.25f }, { 10 });
}

TEST_CASE_FIXTURE(GetSubgraphInputsOutputsFixture, "GetSubgraphOutputsSimpleQuantized")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorIdRawPtrVector subgraphTensors = TfLiteParserImpl::GetSubgraphOutputs(model, 0);
    CHECK_EQ(1, subgraphTensors.size());
    CHECK_EQ(0, subgraphTensors[0].first);
    CheckTensors(subgraphTensors[0].second, 4, { 1, 1, 1, 1 }, tflite::TensorType::TensorType_UINT8, 0,
                      "OutputTensor", { 0.0f }, { 255.0f }, { 1.0f }, { 0 });
}

TEST_CASE_FIXTURE(GetSubgraphInputsOutputsFixture, "GetSubgraphInputsEmptyMinMax")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorIdRawPtrVector subgraphTensors = TfLiteParserImpl::GetSubgraphInputs(model, 1);
    CHECK_EQ(1, subgraphTensors.size());
    CHECK_EQ(0, subgraphTensors[0].first);
    CheckTensors(subgraphTensors[0].second, 4, { 1, 3, 3, 1 }, tflite::TensorType::TensorType_UINT8, 0,
                      "ConvInputTensor", { }, { }, { 1.0f }, { 0 });
}

TEST_CASE_FIXTURE(GetSubgraphInputsOutputsFixture, "GetSubgraphOutputs")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorIdRawPtrVector subgraphTensors = TfLiteParserImpl::GetSubgraphOutputs(model, 1);
    CHECK_EQ(1, subgraphTensors.size());
    CHECK_EQ(1, subgraphTensors[0].first);
    CheckTensors(subgraphTensors[0].second, 4, { 1, 1, 1, 1 }, tflite::TensorType::TensorType_UINT8, 1,
                      "ConvOutputTensor", { 0.0f }, { 511.0f }, { 2.0f }, { 0 });
}

TEST_CASE("GetSubgraphInputsNullModel")
{
    CHECK_THROWS_AS(TfLiteParserImpl::GetSubgraphInputs(nullptr, 0), armnn::ParseException);
}

TEST_CASE("GetSubgraphOutputsNullModel")
{
    CHECK_THROWS_AS(TfLiteParserImpl::GetSubgraphOutputs(nullptr, 0), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetSubgraphInputsOutputsFixture, "GetSubgraphInputsInvalidSubgraph")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetSubgraphInputs(model, 2), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetSubgraphInputsOutputsFixture, "GetSubgraphOutputsInvalidSubgraph")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetSubgraphOutputs(model, 2), armnn::ParseException);
}

}