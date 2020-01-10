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
                                  const std::string& strideX,
                                  const std::string& strideY,
                                  const std::string& dataType)
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "TRANSPOSE_CONV" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 4 ],
                            "type": "UINT8",
                            "buffer": 0,
                            "name": "outputShapeTensor",
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
                        },
                        {
                            "shape": )" + inputShape + R"(,
                            "type": ")" + dataType + R"(",
                            "buffer": 2,
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
                            "type": ")" + dataType + R"(",
                            "buffer": 3,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        }
                    ],
                    "inputs": [ 2 ],
                    "outputs": [ 3 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1, 2 ],
                            "outputs": [ 3 ],
                            "builtin_options_type": "TransposeConvOptions",
                            "builtin_options": {
                                "padding": "VALID",
                                "stride_w": )" + strideX + R"(,
                                "stride_h": )" + strideY + R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { "data": )" + outputShape + R"( },
                    { "data": )" + filterData + R"( },
                    { },
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
                           "1",               // strideX
                           "1",               // strideY
                           "UINT8")           // dataType
    {}
};

BOOST_FIXTURE_TEST_CASE( ParseSimpleTransposeConv, SimpleTransposeConvFixture )
{
    RunTest<4, armnn::DataType::QAsymmU8>(
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

BOOST_AUTO_TEST_SUITE_END()
