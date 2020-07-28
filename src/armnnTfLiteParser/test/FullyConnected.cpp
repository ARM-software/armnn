//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct FullyConnectedFixture : public ParserFlatbuffersFixture
{
    explicit FullyConnectedFixture(const std::string& inputShape,
                                           const std::string& outputShape,
                                           const std::string& filterShape,
                                           const std::string& filterData,
                                           const std::string biasShape = "",
                                           const std::string biasData = "")
    {
        std::string inputTensors = "[ 0, 2 ]";
        std::string biasTensor = "";
        std::string biasBuffer = "";
        if (biasShape.size() > 0 && biasData.size() > 0)
        {
            inputTensors = "[ 0, 2, 3 ]";
            biasTensor = R"(
                        {
                            "shape": )" + biasShape + R"( ,
                            "type": "INT32",
                            "buffer": 3,
                            "name": "biasTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        } )";
            biasBuffer = R"(
                    { "data": )" + biasData + R"(, }, )";
        }
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "FULLY_CONNECTED" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": "UINT8",
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
                            "shape": )" + outputShape + R"(,
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 511.0 ],
                                "scale": [ 2.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + filterShape + R"(,
                            "type": "UINT8",
                            "buffer": 2,
                            "name": "filterTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }, )" + biasTensor + R"(
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": )" + inputTensors + R"(,
                            "outputs": [ 1 ],
                            "builtin_options_type": "FullyConnectedOptions",
                            "builtin_options": {
                                "fused_activation_function": "NONE"
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + filterData + R"(, }, )"
                       + biasBuffer + R"(
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct FullyConnectedWithNoBiasFixture : FullyConnectedFixture
{
    FullyConnectedWithNoBiasFixture()
        : FullyConnectedFixture("[ 1, 4, 1, 1 ]",     // inputShape
                                "[ 1, 1 ]",           // outputShape
                                "[ 1, 4 ]",           // filterShape
                                "[ 2, 3, 4, 5 ]")     // filterData
    {}
};

BOOST_FIXTURE_TEST_CASE(FullyConnectedWithNoBias, FullyConnectedWithNoBiasFixture)
{
    RunTest<2, armnn::DataType::QAsymmU8>(
        0,
        { 10, 20, 30, 40 },
        { 400/2 });
}

struct FullyConnectedWithBiasFixture : FullyConnectedFixture
{
    FullyConnectedWithBiasFixture()
        : FullyConnectedFixture("[ 1, 4, 1, 1 ]",     // inputShape
                                "[ 1, 1 ]",           // outputShape
                                "[ 1, 4 ]",           // filterShape
                                "[ 2, 3, 4, 5 ]",     // filterData
                                "[ 1 ]",              // biasShape
                                "[ 10, 0, 0, 0 ]" )   // biasData
    {}
};

BOOST_FIXTURE_TEST_CASE(ParseFullyConnectedWithBias, FullyConnectedWithBiasFixture)
{
    RunTest<2, armnn::DataType::QAsymmU8>(
        0,
        { 10, 20, 30, 40 },
        { (400+10)/2 });
}

struct FullyConnectedWithBiasMultipleOutputsFixture : FullyConnectedFixture
{
    FullyConnectedWithBiasMultipleOutputsFixture()
            : FullyConnectedFixture("[ 1, 4, 2, 1 ]",     // inputShape
                                    "[ 2, 1 ]",           // outputShape
                                    "[ 1, 4 ]",           // filterShape
                                    "[ 2, 3, 4, 5 ]",     // filterData
                                    "[ 1 ]",              // biasShape
                                    "[ 10, 0, 0, 0 ]" )   // biasData
    {}
};

BOOST_FIXTURE_TEST_CASE(FullyConnectedWithBiasMultipleOutputs, FullyConnectedWithBiasMultipleOutputsFixture)
{
    RunTest<2, armnn::DataType::QAsymmU8>(
            0,
            { 1, 2, 3, 4, 10, 20, 30, 40 },
            { (40+10)/2, (400+10)/2 });
}

struct DynamicFullyConnectedWithBiasMultipleOutputsFixture : FullyConnectedFixture
{
    DynamicFullyConnectedWithBiasMultipleOutputsFixture()
        : FullyConnectedFixture("[ 1, 4, 2, 1 ]",     // inputShape
                                "[ ]",               // outputShape
                                "[ 1, 4 ]",           // filterShape
                                "[ 2, 3, 4, 5 ]",     // filterData
                                "[ 1 ]",              // biasShape
                                "[ 10, 0, 0, 0 ]" )   // biasData
    { }
};

BOOST_FIXTURE_TEST_CASE(
    DynamicFullyConnectedWithBiasMultipleOutputs,
    DynamicFullyConnectedWithBiasMultipleOutputsFixture)
{
    RunTest<2,
            armnn::DataType::QAsymmU8,
            armnn::DataType::QAsymmU8>(0,
                                      { { "inputTensor", { 1, 2, 3, 4, 10, 20, 30, 40} } },
                                      { { "outputTensor", { (40+10)/2, (400+10)/2 } } },
                                      true);
}

BOOST_AUTO_TEST_SUITE_END()
