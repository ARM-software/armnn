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

struct PadFixture : public ParserFlatbuffersFixture
{
    explicit PadFixture(const std::string & inputShape,
                        const std::string & outputShape,
                        const std::string & padListShape,
                        const std::string & padListData)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "PAD" } ],
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
                             "shape": )" + outputShape + R"(,
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
                             "shape": )" + padListShape + R"( ,
                             "type": "INT32",
                             "buffer": 2,
                             "name": "padList",
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
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + padListData + R"(, },
                ]
            }
        )";
      SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimplePadFixture : public PadFixture
{
    SimplePadFixture() : PadFixture("[ 2, 3 ]", "[ 4, 7 ]", "[ 2, 2 ]",
                                    "[  1,0,0,0, 1,0,0,0, 2,0,0,0, 2,0,0,0 ]") {}
};

BOOST_FIXTURE_TEST_CASE(ParsePad, SimplePadFixture)
{
    RunTest<2, armnn::DataType::Float32>
        (0,
         {{ "inputTensor",  { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f }}},
         {{ "outputTensor", { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 1.0f, 2.0f, 3.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 4.0f, 5.0f, 6.0f, 0.0f, 0.0f,
                              0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f }}});
}

BOOST_AUTO_TEST_SUITE_END()
