//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct TransposeConvFixture : public ParserFlatbuffersFixture
{
    explicit TransposeConvFixture(const std::string& inputShape,
                                  const std::string& outputShape,
                                  const std::string& filterShape,
                                  const std::string& filterData,
                                  const bool biasEnabled,
                                  const std::string& biasShape,
                                  const std::string& biasData,
                                  const std::string& strideX,
                                  const std::string& strideY,
                                  const std::string& dataType)
    {
        std::string biasString;
        if (biasEnabled)
        {
            biasString = R"(
                        {
                            "shape": )" + biasShape + R"(,
                            "type": "INT32",
                            "buffer": 2,
                            "name": "biasTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },)";

        }
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "TRANSPOSE_CONV" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": ")" + dataType + R"(",
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
                            "shape": )" + filterShape + R"(,
                            "type": ")" + dataType + R"(",
                            "buffer": 1,
                            "name": "filterTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },)" + biasString + R"(
                        {
                            "shape": )" + outputShape + R"(,
                            "type": ")" + dataType + R"(",
                            "buffer": )" + (biasEnabled ? "3" : "2") + R"(,
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
                    "outputs": [ )" + (biasEnabled ? "3" : "2") + R"( ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1)" + (biasEnabled ? ", 2" : "") + R"( ],
                            "outputs": [ )" + (biasEnabled ? "3" : "2") + R"( ],
                            "builtin_options_type": "TransposeConvOptions",
                            "builtin_options": {
                                "padding": "SAME",
                                "stride_w": )" + strideX + R"(,
                                "stride_h": )" + strideY + R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { "data": )" + filterData + R"( },
                    { )" + (biasEnabled ? (R"("data": )" + biasData) : "") + R"( },
                    { }
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleTransposeConvFixture : TransposeConvFixture
{
    SimpleTransposeConvFixture()
    : TransposeConvFixture("[ 1, 2, 2, 1 ]",  // inputShape
                           "[ 1, 3, 3, 1 ]",  // outputShape
                           "[ 1, 2, 2, 1 ]",  // filterShape
                           "[ 0, 1, 2, 4 ]",  // filterData
                           false,             // biasEnabled
                           "",                // biasShape
                           "",                // biasData
                           "1",               // strideX
                           "1",               // strideY
                           "UINT8")           // dataType
    {}
};

BOOST_FIXTURE_TEST_CASE( ParseSimpleTransposeConv, SimpleTransposeConvFixture )
{
    RunTest<4, armnn::DataType::QuantisedAsymm8>(
        0,
        {
            1, 2,
            3, 4
        },
        {
            0, 1,  2,
            2, 11, 12,
            6, 20, 16
        });
}

struct TransposeConvWithBiasFixture : TransposeConvFixture
{
    TransposeConvWithBiasFixture()
    : TransposeConvFixture("[ 1, 2, 2, 1 ]",  // inputShape
                           "[ 1, 3, 3, 1 ]",  // outputShape
                           "[ 1, 2, 2, 1 ]",  // filterShape
                           "[ 0, 1, 2, 4 ]",  // filterData
                           true,              // biasEnabled
                           "[ 1 ]",           // biasShape
                           "[ 2, 0, 0, 0 ]",  // biasData
                           "1",               // strideX
                           "1",               // strideY
                           "UINT8")           // dataType
    {}
};

BOOST_FIXTURE_TEST_CASE( ParseTransposeConvWithBias, TransposeConvWithBiasFixture )
{
    RunTest<4, armnn::DataType::QuantisedAsymm8>(
        0,
        {
            1, 2,
            3, 4
        },
        {
            2, 3,  5,
            4, 13, 14,
            8, 22, 18
        });
}

BOOST_AUTO_TEST_SUITE_END()
