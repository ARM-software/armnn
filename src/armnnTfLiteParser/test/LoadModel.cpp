//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

#include <armnnUtils/Filesystem.hpp>

using armnnTfLiteParser::TfLiteParserImpl;
using ModelPtr = TfLiteParserImpl::ModelPtr;
using SubgraphPtr = TfLiteParserImpl::SubgraphPtr;
using OperatorPtr = TfLiteParserImpl::OperatorPtr;

TEST_SUITE("TensorflowLiteParser_LoadModel")
{
struct LoadModelFixture : public ParserFlatbuffersFixture
{
    explicit LoadModelFixture()
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
                            "opcode_index": 1,
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
            "description": "Test loading a model",
            "buffers" : [ {}, {} ]
        })";

        ReadStringToBinary();
    }

    void CheckModel(const ModelPtr& model, uint32_t version, size_t opcodeSize,
                    const std::vector<tflite::BuiltinOperator>& opcodes,
                    size_t subgraphs, const std::string desc, size_t buffers)
    {
        CHECK(model);
        CHECK_EQ(version, model->version);
        CHECK_EQ(opcodeSize, model->operator_codes.size());
        CheckBuiltinOperators(opcodes, model->operator_codes);
        CHECK_EQ(subgraphs, model->subgraphs.size());
        CHECK_EQ(desc, model->description);
        CHECK_EQ(buffers, model->buffers.size());
    }

    void CheckBuiltinOperators(const std::vector<tflite::BuiltinOperator>& expectedOperators,
                               const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& result)
    {
        CHECK_EQ(expectedOperators.size(), result.size());
        for (size_t i = 0; i < expectedOperators.size(); i++)
        {
            CHECK_EQ(expectedOperators[i], result[i]->builtin_code);
        }
    }

    void CheckSubgraph(const SubgraphPtr& subgraph, size_t tensors, const std::vector<int32_t>& inputs,
                       const std::vector<int32_t>& outputs, size_t operators, const std::string& name)
    {
        CHECK(subgraph);
        CHECK_EQ(tensors, subgraph->tensors.size());
        CHECK(std::equal(inputs.begin(), inputs.end(), subgraph->inputs.begin(), subgraph->inputs.end()));
        CHECK(std::equal(outputs.begin(), outputs.end(),
                                      subgraph->outputs.begin(), subgraph->outputs.end()));
        CHECK_EQ(operators, subgraph->operators.size());
        CHECK_EQ(name, subgraph->name);
    }

    void CheckOperator(const OperatorPtr& operatorPtr, uint32_t opcode,  const std::vector<int32_t>& inputs,
                       const std::vector<int32_t>& outputs, tflite::BuiltinOptions optionType,
                       tflite::CustomOptionsFormat custom_options_format)
    {
        CHECK(operatorPtr);
        CHECK_EQ(opcode, operatorPtr->opcode_index);
        CHECK(std::equal(inputs.begin(), inputs.end(),
                                      operatorPtr->inputs.begin(), operatorPtr->inputs.end()));
        CHECK(std::equal(outputs.begin(), outputs.end(),
                                      operatorPtr->outputs.begin(), operatorPtr->outputs.end()));
        CHECK_EQ(optionType, operatorPtr->builtin_options.type);
        CHECK_EQ(custom_options_format, operatorPtr->custom_options_format);
    }
};

TEST_CASE_FIXTURE(LoadModelFixture, "LoadModelFromBinary")
{
    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromBinary(m_GraphBinary.data(),
                                                                             m_GraphBinary.size());
    CheckModel(model, 3, 2, { tflite::BuiltinOperator_AVERAGE_POOL_2D, tflite::BuiltinOperator_CONV_2D },
               2, "Test loading a model", 2);
    CheckSubgraph(model->subgraphs[0], 2, { 1 }, { 0 }, 1, "");
    CheckSubgraph(model->subgraphs[1], 3, { 0 }, { 1 }, 1, "");
    CheckOperator(model->subgraphs[0]->operators[0], 0, { 1 }, { 0 }, tflite::BuiltinOptions_Pool2DOptions,
                  tflite::CustomOptionsFormat_FLEXBUFFERS);
    CheckOperator(model->subgraphs[1]->operators[0], 1, { 0, 2 }, { 1 }, tflite::BuiltinOptions_Conv2DOptions,
                  tflite::CustomOptionsFormat_FLEXBUFFERS);
}

TEST_CASE_FIXTURE(LoadModelFixture, "LoadModelFromFile")
{
    using namespace fs;
    fs::path fname = armnnUtils::Filesystem::NamedTempFile("Armnn-tfLite-LoadModelFromFile-TempFile.csv");
    bool saved = flatbuffers::SaveFile(fname.c_str(),
                                       reinterpret_cast<char *>(m_GraphBinary.data()),
                                       m_GraphBinary.size(), true);
    CHECK_MESSAGE(saved, "Cannot save test file");

    TfLiteParserImpl::ModelPtr model = TfLiteParserImpl::LoadModelFromFile(fname.c_str());
    CheckModel(model, 3, 2, { tflite::BuiltinOperator_AVERAGE_POOL_2D, tflite::BuiltinOperator_CONV_2D },
               2, "Test loading a model", 2);
    CheckSubgraph(model->subgraphs[0], 2, { 1 }, { 0 }, 1, "");
    CheckSubgraph(model->subgraphs[1], 3, { 0 }, { 1 }, 1, "");
    CheckOperator(model->subgraphs[0]->operators[0], 0, { 1 }, { 0 }, tflite::BuiltinOptions_Pool2DOptions,
                  tflite::CustomOptionsFormat_FLEXBUFFERS);
    CheckOperator(model->subgraphs[1]->operators[0], 1, { 0, 2 }, { 1 }, tflite::BuiltinOptions_Conv2DOptions,
                  tflite::CustomOptionsFormat_FLEXBUFFERS);
    remove(fname);
}

TEST_CASE("LoadNullBinary")
{
    CHECK_THROWS_AS(TfLiteParserImpl::LoadModelFromBinary(nullptr, 0), armnn::InvalidArgumentException);
}

TEST_CASE("LoadInvalidBinary")
{
    std::string testData = "invalid data";
    CHECK_THROWS_AS(TfLiteParserImpl::LoadModelFromBinary(reinterpret_cast<const uint8_t*>(&testData),
                                                        testData.length()), armnn::ParseException);
}

TEST_CASE("LoadFileNotFound")
{
    CHECK_THROWS_AS(TfLiteParserImpl::LoadModelFromFile("invalidfile.tflite"), armnn::FileNotFoundException);
}

TEST_CASE("LoadNullPtrFile")
{
    CHECK_THROWS_AS(TfLiteParserImpl::LoadModelFromFile(nullptr), armnn::InvalidArgumentException);
}

}
