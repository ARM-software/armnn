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

struct UnpackFixture : public ParserFlatbuffersFixture
{
    explicit UnpackFixture(const std::string & inputShape,
                           const unsigned int numberOfOutputs,
                           const std::string & outputShape,
                           const std::string & axis,
                           const std::string & num)
    {
        // As input index is 0, output indexes start at 1
        std::string outputIndexes = "1";
        for(unsigned int i = 1; i < numberOfOutputs; i++)
        {
            outputIndexes += ", " + std::to_string(i+1);
        }
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "UNPACK" } ],
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
                        },)";
        // Append the required number of outputs for this UnpackFixture.
        // As input index is 0, output indexes start at 1.
        for(unsigned int i = 0; i < numberOfOutputs; i++)
        {
            m_JsonString += R"(
                        {
                            "shape": )" + outputShape + R"( ,
                                "type": "FLOAT32",
                                "buffer": )" + std::to_string(i + 1) + R"(,
                                "name": "outputTensor)" + std::to_string(i + 1) + R"(",
                                "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },)";
        }
        m_JsonString += R"(
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ )" + outputIndexes + R"( ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0 ],
                            "outputs": [ )" + outputIndexes + R"( ],
                            "builtin_options_type": "UnpackOptions",
                            "builtin_options": {
                                "axis": )" + axis;

                    if(!num.empty())
                    {
                        m_JsonString += R"(,
                                "num" : )" + num;
                    }

                    m_JsonString += R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { }
                ]
            }
        )";
        Setup();
    }
};

struct DefaultUnpackAxisZeroFixture : UnpackFixture
{
    DefaultUnpackAxisZeroFixture() : UnpackFixture("[ 4, 1, 6 ]", 4, "[ 1, 6 ]", "0", "") {}
};

BOOST_FIXTURE_TEST_CASE(UnpackAxisZeroNumIsDefaultNotSpecified, DefaultUnpackAxisZeroFixture)
{
    RunTest<2, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                            19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f } } },
        { {"outputTensor1", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }},
          {"outputTensor2", { 7.0f,  8.0f,  9.0f, 10.0f, 11.0f, 12.0f }},
          {"outputTensor3", { 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f }},
          {"outputTensor4", { 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f }} });
}

struct DefaultUnpackLastAxisFixture : UnpackFixture
{
    DefaultUnpackLastAxisFixture() : UnpackFixture("[ 4, 1, 6 ]", 6, "[ 4, 1 ]", "2", "6") {}
};

BOOST_FIXTURE_TEST_CASE(UnpackLastAxisNumSix, DefaultUnpackLastAxisFixture)
{
    RunTest<2, armnn::DataType::Float32>(
        0,
        { {"inputTensor", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                            7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f,
                            13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
                            19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f } } },
        { {"outputTensor1", { 1.0f, 7.0f, 13.0f, 19.0f }},
          {"outputTensor2", { 2.0f, 8.0f, 14.0f, 20.0f }},
          {"outputTensor3", { 3.0f, 9.0f, 15.0f, 21.0f }},
          {"outputTensor4", { 4.0f, 10.0f, 16.0f, 22.0f }},
          {"outputTensor5", { 5.0f, 11.0f, 17.0f, 23.0f }},
          {"outputTensor6", { 6.0f, 12.0f, 18.0f, 24.0f }} });
}

BOOST_AUTO_TEST_SUITE_END()
