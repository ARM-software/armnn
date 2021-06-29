//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

using armnnTfLiteParser::TfLiteParserImpl;
using ModelPtr = TfLiteParserImpl::ModelPtr;

TEST_SUITE("TensorflowLiteParser_GetInputsOutputs")
{
struct GetInputsOutputsMainFixture : public ParserFlatbuffersFixture
{
    explicit GetInputsOutputsMainFixture(const std::string& inputs, const std::string& outputs)
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
                "inputs": [ 1 ],
                "outputs": [ 0 ],
                "operators": [ {
                        "opcode_index": 0,
                        "inputs":  )"
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

struct GetEmptyInputsOutputsFixture : GetInputsOutputsMainFixture
{
    GetEmptyInputsOutputsFixture() : GetInputsOutputsMainFixture("[ ]", "[ ]") {}
};

struct GetInputsOutputsFixture : GetInputsOutputsMainFixture
{
    GetInputsOutputsFixture() : GetInputsOutputsMainFixture("[ 1 ]", "[ 0 ]") {}
};

TEST_CASE_FIXTURE(GetEmptyInputsOutputsFixture, "GetEmptyInputs")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorRawPtrVector tensors = TfLiteParserImpl::GetInputs(model, 0, 0);
    CHECK_EQ(0, tensors.size());
}

TEST_CASE_FIXTURE(GetEmptyInputsOutputsFixture, "GetEmptyOutputs")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorRawPtrVector tensors = TfLiteParserImpl::GetOutputs(model, 0, 0);
    CHECK_EQ(0, tensors.size());
}

TEST_CASE_FIXTURE(GetInputsOutputsFixture, "GetInputs")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorRawPtrVector tensors = TfLiteParserImpl::GetInputs(model, 0, 0);
    CHECK_EQ(1, tensors.size());
    CheckTensors(tensors[0], 4, { 1, 2, 2, 1 }, tflite::TensorType::TensorType_UINT8, 1,
                      "InputTensor", { -1.2f }, { 25.5f }, { 0.25f }, { 10 });
}

TEST_CASE_FIXTURE(GetInputsOutputsFixture, "GetOutputs")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorRawPtrVector tensors = TfLiteParserImpl::GetOutputs(model, 0, 0);
    CHECK_EQ(1, tensors.size());
    CheckTensors(tensors[0], 4, { 1, 1, 1, 1 }, tflite::TensorType::TensorType_UINT8, 0,
                      "OutputTensor", { 0.0f }, { 255.0f }, { 1.0f }, { 0 });
}

TEST_CASE_FIXTURE(GetInputsOutputsFixture, "GetInputsMultipleInputs")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorRawPtrVector tensors = TfLiteParserImpl::GetInputs(model, 1, 0);
    CHECK_EQ(2, tensors.size());
    CheckTensors(tensors[0], 4, { 1, 3, 3, 1 }, tflite::TensorType::TensorType_UINT8, 0,
                      "ConvInputTensor", { }, { }, { 1.0f }, { 0 });
    CheckTensors(tensors[1], 4, { 1, 3, 3, 1 }, tflite::TensorType::TensorType_UINT8, 2,
                      "filterTensor", { 0.0f }, { 255.0f }, { 1.0f }, { 0 });
}

TEST_CASE_FIXTURE(GetInputsOutputsFixture, "GetOutputs2")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    TfLiteParserImpl::TensorRawPtrVector tensors = TfLiteParserImpl::GetOutputs(model, 1, 0);
    CHECK_EQ(1, tensors.size());
    CheckTensors(tensors[0], 4, { 1, 1, 1, 1 }, tflite::TensorType::TensorType_UINT8, 1,
                      "ConvOutputTensor", { 0.0f }, { 511.0f }, { 2.0f }, { 0 });
}

TEST_CASE("GetInputsNullModel")
{
    CHECK_THROWS_AS(TfLiteParserImpl::GetInputs(nullptr, 0, 0), armnn::ParseException);
}

TEST_CASE("GetOutputsNullModel")
{
    CHECK_THROWS_AS(TfLiteParserImpl::GetOutputs(nullptr, 0, 0), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetInputsOutputsFixture, "GetInputsInvalidSubgraph")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetInputs(model, 2, 0), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetInputsOutputsFixture, "GetOutputsInvalidSubgraph")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetOutputs(model, 2, 0), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetInputsOutputsFixture, "GetInputsInvalidOperator")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetInputs(model, 0, 1), armnn::ParseException);
}

TEST_CASE_FIXTURE(GetInputsOutputsFixture, "GetOutputsInvalidOperator")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CHECK_THROWS_AS(TfLiteParserImpl::GetOutputs(model, 0, 1), armnn::ParseException);
}

}