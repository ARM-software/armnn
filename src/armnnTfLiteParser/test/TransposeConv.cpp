//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"

TEST_SUITE("TensorflowLiteParser_TransposeConv")
{
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

TEST_CASE_FIXTURE(SimpleTransposeConvFixture, "ParseSimpleTransposeConv")
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

struct TransposeConvFixtureWithBias : public ParserFlatbuffersFixture
{
    explicit TransposeConvFixtureWithBias(const std::string& inputShape,
                                          const std::string& outputShape,
                                          const std::string& filterShape,
                                          const std::string& filterData,
                                          const std::string& strideX,
                                          const std::string& strideY,
                                          const std::string& dataType,
                                          const std::string& biasShape,
                                          const std::string& biasData)
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
                        },
                        {
                            "shape": )" + outputShape + R"(,
                            "type": ")" + dataType + R"(",
                            "buffer": 4,
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
                    "outputs": [ 4 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 1, 2, 3],
                            "outputs": [ 4 ],
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
                    { "data": )" + biasData + R"( },
                    { }
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleTransposeConvFixtureWithBias : TransposeConvFixtureWithBias
{
    SimpleTransposeConvFixtureWithBias()
    : TransposeConvFixtureWithBias("[ 1, 2, 2, 1 ]",  // inputShape
                                   "[ 1, 3, 3, 1 ]",  // outputShape
                                   "[ 1, 2, 2, 1 ]",  // filterShape
                                   "[ 0, 1, 2, 4 ]",  // filterData
                                   "1",               // strideX
                                   "1",               // strideY
                                   "UINT8",           // dataType
                                   "[ 1 ]",           // bias shape
                                   "[ 10, 0, 0, 0 ]") // bias data
    {}
};

TEST_CASE_FIXTURE(SimpleTransposeConvFixtureWithBias, "ParseSimpleTransposeConvWithBias")
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        {
            1, 2,
            3, 4
        },
        {
            10, 11, 12,
            12, 21, 22,
            16, 30, 26
        });
}


struct TransposeConvPerChannelFixture : public ParserFlatbuffersFixture
{
    explicit TransposeConvPerChannelFixture()
    {
        m_JsonString = R"(
        {
            "version": 3,
            "operator_codes": [
                {
                    "builtin_code": "TRANSPOSE_CONV",
                    "version": 2
                }
            ],
            "subgraphs": [
                {
                    "tensors": [
                        {
                            "shape": [
                                1,
                                4,
                                4,
                                2
                            ],
                            "type": "INT8",
                            "buffer": 1,
                            "name": "input",
                            "quantization": {
                                "min": [
                                    -50.0
                                ],
                                "max": [
                                    49.0
                                ],
                                "scale": [
                                    0.388235
                                ],
                                "zero_point": [
                                    1
                                ],
                                "details_type": "NONE",
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },
                        {
                            "shape": [
                                4
                            ],
                            "type": "INT32",
                            "buffer": 2,
                            "name": "model/conv2d_transpose/stack",
                            "quantization": {
                                "details_type": "NONE",
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },
                        {
                            "shape": [
                                8,
                                2,
                                2,
                                2
                            ],
                            "type": "INT8",
                            "buffer": 3,
                            "name": "model/conv2d_transpose/conv2d_transpose",
                            "quantization": {
                                "min": [
                                    -0.081948,
                                    -0.379918,
                                    -0.223632,
                                    -0.098629,
                                    -0.386369,
                                    -0.351057,
                                    -0.348749,
                                    -0.264848
                                ],
                                "max": [
                                    0.35091,
                                    0.229681,
                                    0.368384,
                                    0.176761,
                                    0.353717,
                                    0.377565,
                                    0.373713,
                                    0.30141
                                ],
                                "scale": [
                                    0.002763,
                                    0.002991,
                                    0.002901,
                                    0.001392,
                                    0.003042,
                                    0.002973,
                                    0.002943,
                                    0.002373
                                ],
                                "zero_point": [
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0,
                                    0
                                ],
                                "details_type": "NONE",
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        },
                        {
                            "shape": [
                                1,
                                4,
                                4,
                                8
                            ],
                            "type": "INT8",
                            "buffer": 4,
                            "name": "Identity",
                            "quantization": {
                                "min": [
                                    -63.578175
                                ],
                                "max": [
                                    69.305023
                                ],
                                "scale": [
                                    0.521111
                                ],
                                "zero_point": [
                                    -6
                                ],
                                "details_type": "NONE",
                                "quantized_dimension": 0
                            },
                            "is_variable": false
                        }
                    ],
                    "inputs": [
                        0
                    ],
                    "outputs": [
                        3
                    ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [
                                1,
                                2,
                                0
                            ],
                            "outputs": [
                                3
                            ],
                            "builtin_options_type": "TransposeConvOptions",
                            "builtin_options": {
                                "padding": "SAME",
                                "stride_w": 1,
                                "stride_h": 1
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                    "name": "main"
                }
            ],
            "description": "MLIR Converted.",
            "buffers": [
                {
                },
                {
                },
                {
                    "data": [
                        1,
                        0,
                        0,
                        0,
                        4,
                        0,
                        0,
                        0,
                        4,
                        0,
                        0,
                        0,
                        8,
                        0,
                        0,
                        0
                    ]
                },
                {
                    "data": [
                        13,
                        239,
                        7,
                        125,
                        35,
                        127,
                        55,
                        226,
                        77,
                        150,
                        159,
                        192,
                        180,
                        129,
                        51,
                        48,
                        108,
                        9,
                        21,
                        179,
                        12,
                        39,
                        127,
                        107,
                        44,
                        206,
                        127,
                        185,
                        108,
                        82,
                        86,
                        218,
                        38,
                        149,
                        16,
                        1,
                        129,
                        163,
                        116,
                        136,
                        138,
                        43,
                        65,
                        186,
                        154,
                        138,
                        64,
                        127,
                        120,
                        127,
                        207,
                        70,
                        43,
                        33,
                        141,
                        137,
                        93,
                        215,
                        65,
                        92,
                        122,
                        144,
                        120,
                        127
                    ]
                },
                {
                },
                {
                    "data": [
                        49,
                        46,
                        57,
                        46,
                        48,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0
                    ]
                }
              ],
            "metadata": [
                {
                    "name": "min_runtime_version",
                    "buffer": 5
                }
            ]
        }
        )";
        SetupSingleInputSingleOutput("input", "Identity");
    }
};

TEST_CASE_FIXTURE(TransposeConvPerChannelFixture, "ParseTransposeConvPerChannel")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        {
            -11, 40,-26, 11,-28,  8,  0, -8,
            -10, 34, 47,  0,-33,-14, 28, 35,
              6,-28,-26,  8, 13, 33,-31,-41,
             31,-20,-31,-16,  8,-18,-44,  0
        },
        {
            -8,-17, -8, -9,-16,  1,  2,-11,
             3,-16,-19,-12,-11, -6, -3, -6,
            -5, -8,-16,-12,-11, -3, -7,-13,
            -4,  1, -9,-10, -5,-12, -5, -8,
             2,-25, -5, -6,-20, -7,  2,-21,
             1,  4,  5,-13,-10,-12,  3,  4,
           -10,-17,-17, -6, -7, 12,-22,-17,
           -17,  0, -5,-14,-21,-12, 17,-13,
             3, -6, -3, -3, -2,-16,-11,-12,
           -15,-14, -1, -2,-35,  5,-18,  0,
            -6,  8,  5,-12, 12,  7, -6, -3,
            11,-28,-28, -3,-18,-29, -5,-13,
           -12, 11, -2, -5,  6, -9, -6,  7,
            -9,-11,-14, -2, 12,  5,-21,-23,
            -4, -4, -6, -6,-21,-25,  0,-18,
           -26, 10, -7,-13,  3, 39,-39, -4
        });
}

}
