//
// Copyright © 2019 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_Quantize")
{
    struct QuantizeFixture : public ParserFlatbuffersFixture
    {
        explicit QuantizeFixture(const std::string& inputShape,
                                 const std::string& outputShape,
                                 const std::string& dataType,
                                 const std::string& zeroPoint = "[ 0 ]")
        {
            m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "QUANTIZE" } ],
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
                                "zero_point": )" + zeroPoint + R"(,
                            }
                        },
                        {
                            "shape": )" + outputShape + R"( ,
                            "type": )" + dataType + R"(,
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.5 ],
                                "zero_point": )" + zeroPoint + R"(,
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
                            "builtin_options_type": "QuantizeOptions",
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

    struct SimpleQuantizeFixtureQAsymm8 : QuantizeFixture
    {
        SimpleQuantizeFixtureQAsymm8() : QuantizeFixture("[ 1, 6 ]",
                                                         "[ 1, 6 ]",
                                                         "UINT8") {}
    };

    TEST_CASE_FIXTURE(SimpleQuantizeFixtureQAsymm8, "SimpleQuantizeFixtureQAsymm8")
    {
        RunTest<2, armnn::DataType::Float32, armnn::DataType::QAsymmU8>(
                0,
                {{"inputTensor",  { 0.0f, 1.5f, 7.5f, 150.0f, 300.0f, 382.5f }}},
                {{"outputTensor", { 0u,   1u,   5u,   100u,   200u,   255u }}});
    }

    struct SimpleQuantizeFixtureQSymm16 : QuantizeFixture
    {
        SimpleQuantizeFixtureQSymm16() : QuantizeFixture("[ 1, 6 ]",
                                                         "[ 1, 6 ]",
                                                         "INT16") {}
    };

    TEST_CASE_FIXTURE(SimpleQuantizeFixtureQSymm16, "SimpleQuantizeQsymm16")
    {
        RunTest<2, armnn::DataType::Float32, armnn::DataType::QSymmS16>(
                0,
                {{"inputTensor",  { 0.0f, 1.5f, 7.5f, 49150.5f, -1.5f,-49152.0f }}},
                {{"outputTensor", { 0,    1,    5,    32767,    -1,   -32768 }}});
    }

    struct SimpleQuantizeFixtureQSymmS8 : QuantizeFixture
    {
        SimpleQuantizeFixtureQSymmS8() : QuantizeFixture("[ 1, 6 ]",
                                                         "[ 1, 6 ]",
                                                         "INT8",
                                                         "[]") {}
    };

    TEST_CASE_FIXTURE(SimpleQuantizeFixtureQSymmS8, "SimpleQuantizeQSymmS8")
    {
        RunTest<2, armnn::DataType::Float32, armnn::DataType::QSymmS8>(
                0,
                {{"inputTensor",  { 0.0f, 1.5f, 7.5f, 190.5f, -192.0f, -1.5f }}},
                {{"outputTensor", { 0,    1,    5,    127,    -128,   -1 }}});
    }

}
