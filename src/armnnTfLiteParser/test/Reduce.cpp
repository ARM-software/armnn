//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct ReduceMaxFixture : public ParserFlatbuffersFixture
{
    explicit ReduceMaxFixture(const std::string& inputShape,
                              const std::string& outputShape,
                              const std::string& axisShape,
                              const std::string& axisData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "REDUCE_MAX" } ],
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
                            "shape": )" + outputShape + R"( ,
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
                            "shape": )" + axisShape + R"( ,
                            "type": "INT32",
                            "buffer": 2,
                            "name": "axis",
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
                            "inputs": [ 0 , 2 ],
                            "outputs": [ 1 ],
                            "builtin_options_type": "ReducerOptions",
                            "builtin_options": {
                              "keep_dims": true,
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + axisData + R"(, },
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleReduceMaxFixture : public ReduceMaxFixture
{
    SimpleReduceMaxFixture() : ReduceMaxFixture("[ 1, 1, 2, 3 ]", "[ 1, 1, 1, 3 ]", "[ 1 ]", "[ 2 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseReduceMax, SimpleReduceMaxFixture)
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Float32>
        (0, {{ "inputTensor",  { 1001.0f, 11.0f,   1003.0f,
                                 10.0f,   1002.0f, 12.0f } } },
            {{ "outputTensor", { 1001.0f, 1002.0f, 1003.0f } } });
}

struct ReduceMinFixture : public ParserFlatbuffersFixture
{
    explicit ReduceMinFixture(const std::string& inputShape,
                              const std::string& outputShape,
                              const std::string& axisShape,
                              const std::string& axisData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "REDUCE_MIN" } ],
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
                            "shape": )" + outputShape + R"( ,
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
                            "shape": )" + axisShape + R"( ,
                            "type": "INT32",
                            "buffer": 2,
                            "name": "axis",
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
                            "inputs": [ 0 , 2 ],
                            "outputs": [ 1 ],
                            "builtin_options_type": "ReducerOptions",
                            "builtin_options": {
                              "keep_dims": true,
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + axisData + R"(, },
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleReduceMinFixture : public ReduceMinFixture
{
    SimpleReduceMinFixture() : ReduceMinFixture("[ 1, 1, 2, 3 ]", "[ 1, 1, 1, 3 ]", "[ 1 ]", "[ 2 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParseReduceMin, SimpleReduceMinFixture)
{
    RunTest<4, armnn::DataType::Float32, armnn::DataType::Float32>
        (0, {{ "inputTensor",  { 1001.0f, 11.0f,   1003.0f,
                                 10.0f,   1002.0f, 12.0f } } },
            {{ "outputTensor", { 10.0f, 11.0f, 12.0f } } });
}

BOOST_AUTO_TEST_SUITE_END()
