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

struct SoftmaxFixture : public ParserFlatbuffersFixture
{
    explicit SoftmaxFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "SOFTMAX" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 1, 7 ],
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
                            "shape": [ 1, 7 ],
                            "type": "UINT8",
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 0.00390625 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                          "opcode_index": 0,
                          "inputs": [ 0 ],
                          "outputs": [ 1 ],
                          "builtin_options_type": "SoftmaxOptions",
                          "builtin_options": {
                            "beta": 1.0
                          },
                          "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [ {}, {} ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

BOOST_FIXTURE_TEST_CASE(ParseSoftmaxLite, SoftmaxFixture)
{
    RunTest<2, armnn::DataType::QAsymmU8>(0, { 0, 0, 100, 0, 0, 0, 0 }, { 0, 0, 255, 0, 0, 0, 0 });
}

BOOST_AUTO_TEST_SUITE_END()
