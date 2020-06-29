//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//
#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <Filesystem.hpp>

using armnnTfLiteParser::TfLiteParser;
using ModelPtr = TfLiteParser::ModelPtr;
using SubgraphPtr = TfLiteParser::SubgraphPtr;
using OperatorPtr = TfLiteParser::OperatorPtr;

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

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
        BOOST_CHECK(model);
        BOOST_CHECK_EQUAL(version, model->version);
        BOOST_CHECK_EQUAL(opcodeSize, model->operator_codes.size());
        CheckBuiltinOperators(opcodes, model->operator_codes);
        BOOST_CHECK_EQUAL(subgraphs, model->subgraphs.size());
        BOOST_CHECK_EQUAL(desc, model->description);
        BOOST_CHECK_EQUAL(buffers, model->buffers.size());
    }

    void CheckBuiltinOperators(const std::vector<tflite::BuiltinOperator>& expectedOperators,
                               const std::vector<std::unique_ptr<tflite::OperatorCodeT>>& result)
    {
        BOOST_CHECK_EQUAL(expectedOperators.size(), result.size());
        for (size_t i = 0; i < expectedOperators.size(); i++)
        {
            BOOST_CHECK_EQUAL(expectedOperators[i], result[i]->builtin_code);
        }
    }

    void CheckSubgraph(const SubgraphPtr& subgraph, size_t tensors, const std::vector<int32_t>& inputs,
                       const std::vector<int32_t>& outputs, size_t operators, const std::string& name)
    {
        BOOST_CHECK(subgraph);
        BOOST_CHECK_EQUAL(tensors, subgraph->tensors.size());
        BOOST_CHECK_EQUAL_COLLECTIONS(inputs.begin(), inputs.end(), subgraph->inputs.begin(), subgraph->inputs.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(outputs.begin(), outputs.end(),
                                      subgraph->outputs.begin(), subgraph->outputs.end());
        BOOST_CHECK_EQUAL(operators, subgraph->operators.size());
        BOOST_CHECK_EQUAL(name, subgraph->name);
    }

    void CheckOperator(const OperatorPtr& operatorPtr, uint32_t opcode,  const std::vector<int32_t>& inputs,
                       const std::vector<int32_t>& outputs, tflite::BuiltinOptions optionType,
                       tflite::CustomOptionsFormat custom_options_format)
    {
        BOOST_CHECK(operatorPtr);
        BOOST_CHECK_EQUAL(opcode, operatorPtr->opcode_index);
        BOOST_CHECK_EQUAL_COLLECTIONS(inputs.begin(), inputs.end(),
                                      operatorPtr->inputs.begin(), operatorPtr->inputs.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(outputs.begin(), outputs.end(),
                                      operatorPtr->outputs.begin(), operatorPtr->outputs.end());
        BOOST_CHECK_EQUAL(optionType, operatorPtr->builtin_options.type);
        BOOST_CHECK_EQUAL(custom_options_format, operatorPtr->custom_options_format);
    }
};

BOOST_FIXTURE_TEST_CASE(LoadModelFromBinary, LoadModelFixture)
{
    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromBinary(m_GraphBinary.data(), m_GraphBinary.size());
    CheckModel(model, 3, 2, { tflite::BuiltinOperator_AVERAGE_POOL_2D, tflite::BuiltinOperator_CONV_2D },
               2, "Test loading a model", 2);
    CheckSubgraph(model->subgraphs[0], 2, { 1 }, { 0 }, 1, "");
    CheckSubgraph(model->subgraphs[1], 3, { 0 }, { 1 }, 1, "");
    CheckOperator(model->subgraphs[0]->operators[0], 0, { 1 }, { 0 }, tflite::BuiltinOptions_Pool2DOptions,
                  tflite::CustomOptionsFormat_FLEXBUFFERS);
    CheckOperator(model->subgraphs[1]->operators[0], 1, { 0, 2 }, { 1 }, tflite::BuiltinOptions_Conv2DOptions,
                  tflite::CustomOptionsFormat_FLEXBUFFERS);
}

BOOST_FIXTURE_TEST_CASE(LoadModelFromFile, LoadModelFixture)
{
    using namespace fs;
    fs::path fname = armnnUtils::Filesystem::NamedTempFile("Armnn-tfLite-LoadModelFromFile-TempFile.csv");
    bool saved = flatbuffers::SaveFile(fname.c_str(),
                                       reinterpret_cast<char *>(m_GraphBinary.data()),
                                       m_GraphBinary.size(), true);
    BOOST_CHECK_MESSAGE(saved, "Cannot save test file");

    TfLiteParser::ModelPtr model = TfLiteParser::LoadModelFromFile(fname.c_str());
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

BOOST_AUTO_TEST_CASE(LoadNullBinary)
{
    BOOST_CHECK_THROW(TfLiteParser::LoadModelFromBinary(nullptr, 0), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_CASE(LoadInvalidBinary)
{
    std::string testData = "invalid data";
    BOOST_CHECK_THROW(TfLiteParser::LoadModelFromBinary(reinterpret_cast<const uint8_t*>(&testData),
                                                        testData.length()), armnn::ParseException);
}

BOOST_AUTO_TEST_CASE(LoadFileNotFound)
{
    BOOST_CHECK_THROW(TfLiteParser::LoadModelFromFile("invalidfile.tflite"), armnn::FileNotFoundException);
}

BOOST_AUTO_TEST_CASE(LoadNullPtrFile)
{
    BOOST_CHECK_THROW(TfLiteParser::LoadModelFromFile(nullptr), armnn::InvalidArgumentException);
}

BOOST_AUTO_TEST_SUITE_END()
