//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

#include <string>
#include <iostream>

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

    struct DequantizeFixture : public ParserFlatbuffersFixture
    {
        explicit DequantizeFixture(const std::string & inputShape,
                                   const std::string & outputShape,
                                   const std::string & dataType)
        {
            m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "DEQUANTIZE" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": )" + dataType + R"(,
                            "buffer": 0,
                            "name": "inputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.5 ],
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
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0 ],
                            "outputs": [ 1 ],
                            "builtin_options_type": "DequantizeOptions",
                            "builtin_options": {
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                ],
                } ],
                "buffers" : [
                    { },
                    { },
                ]
            }
        )";
            SetupSingleInputSingleOutput("inputTensor", "outputTensor");
        }
    };

    struct SimpleDequantizeFixtureQAsymm8 : DequantizeFixture
    {
        SimpleDequantizeFixtureQAsymm8() : DequantizeFixture("[ 1, 6 ]",
                                                             "[ 1, 6 ]",
                                                             "UINT8") {}
    };

    BOOST_FIXTURE_TEST_CASE(SimpleDequantizeQAsymm8, SimpleDequantizeFixtureQAsymm8)
    {
        RunTest<2, armnn::DataType::QAsymmU8 , armnn::DataType::Float32>(
                0,
                {{"inputTensor",  { 0u,   1u,   5u,   100u,   200u,   255u }}},
                {{"outputTensor", { 0.0f, 1.5f, 7.5f, 150.0f, 300.0f, 382.5f }}});
    }

    struct SimpleDequantizeFixtureQSymm16 : DequantizeFixture
    {
        SimpleDequantizeFixtureQSymm16() : DequantizeFixture("[ 1, 6 ]",
                                                             "[ 1, 6 ]",
                                                             "INT16") {}
    };

    BOOST_FIXTURE_TEST_CASE(SimpleDequantizeQsymm16, SimpleDequantizeFixtureQSymm16)
    {
        RunTest<2, armnn::DataType::QSymmS16 , armnn::DataType::Float32>(
                0,
                {{"inputTensor",  { 0,    1,    5,    32767,    -1,   -32768 }}},
                {{"outputTensor", { 0.0f, 1.5f, 7.5f, 49150.5f, -1.5f,-49152.0f }}});
    }

    struct SimpleDequantizeFixtureQAsymmS8 : DequantizeFixture
    {
        SimpleDequantizeFixtureQAsymmS8() : DequantizeFixture("[ 1, 6 ]",
                                                             "[ 1, 6 ]",
                                                             "INT8") {}
    };

    BOOST_FIXTURE_TEST_CASE(SimpleDequantizeQAsymmS8, SimpleDequantizeFixtureQAsymmS8)
    {
        RunTest<2, armnn::DataType::QAsymmS8 , armnn::DataType::Float32>(
                0,
                {{"inputTensor",  { 0,    1,    5,    127,    -128,   -1 }}},
                {{"outputTensor", { 0.0f, 1.5f, 7.5f, 190.5f, -192.0f, -1.5f }}});
    }

BOOST_AUTO_TEST_SUITE_END()
