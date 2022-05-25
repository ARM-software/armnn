//
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"
#include <sstream>

// Conv3D support was added in TF 2.5, so for backwards compatibility a hash define is needed.
#if defined(ARMNN_POST_TFLITE_2_4)
TEST_SUITE("TensorflowLiteParser_Conv3D")
{
struct SimpleConv3DFixture : public ParserFlatbuffersFixture
{
    explicit SimpleConv3DFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "CONV_3D" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 1, 2, 3, 3, 1 ],
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
                            "shape": [ 1, 1, 1, 1, 1 ],
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
                            "shape": [ 2, 3, 3, 1, 1 ],
                            "type": "UINT8",
                            "buffer": 2,
                            "name": "filterTensor",
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
                            "builtin_options_type": "Conv3DOptions",
                            "builtin_options": {
                                "padding": "VALID",
                                "stride_d": 1,
                                "stride_w": 1,
                                "stride_h": 1,
                                "fused_activation_function": "NONE"
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": [ 2,1,0,  6,2,1, 4,1,2,
                                1,2,1,  2,0,2, 2,1,1 ], },
                    { },
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

TEST_CASE_FIXTURE(SimpleConv3DFixture, "ParseSimpleConv3D")
{
    RunTest<5, armnn::DataType::QAsymmU8>(
        0,
        {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,

            10, 11, 12,
            13, 14, 15,
            16, 17, 18,
        },
        // Due to the output scaling we need to half the values.
        {
            (1*2 + 2*1 + 3*0 +
             4*6 + 5*2 + 6*1 +
             7*4 + 8*1 + 9*2 +

             10*1 + 11*2 + 12*1 +
             13*2 + 14*0 + 15*2 +
             16*2 + 17*1 + 18*1) /2
        });
}
struct Conv3DWithBiasesFixture : public ParserFlatbuffersFixture
{
    explicit Conv3DWithBiasesFixture(const std::string& inputShape,
                                     const std::string& outputShape,
                                     const std::string& filterShape,
                                     const std::string& filterData,
                                     const std::string& biasShape,
                                     const std::string& biasData,
                                     const std::string& strides,
                                     const std::string& activation="NONE",
                                     const std::string& filterScale="1.0",
                                     const std::string& filterZeroPoint="0",
                                     const std::string& outputScale="1.0",
                                     const std::string& outputZeroPoint="0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "CONV_3D" } ],
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
                                "scale": [ )" + outputScale + R"( ],
                                "zero_point": [ )" + outputZeroPoint + R"( ],
                            }
                        },
                        {
                            "shape": )" + filterShape + R"( ,
                            "type": "UINT8",
                            "buffer": 2,
                            "name": "filterTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 255.0 ],
                                "scale": [ )" + filterScale + R"( ],
                                "zero_point": [ )" + filterZeroPoint + R"( ],
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
                        }
                    ],
                    "inputs": [ 0 ],
                    "outputs": [ 1 ],
                    "operators": [
                        {
                            "opcode_index": 0,
                            "inputs": [ 0, 2, 3 ],
                            "outputs": [ 1 ],
                            "builtin_options_type": "Conv3DOptions",
                            "builtin_options": {
                                "padding": "SAME",
                                "stride_d": )" + strides + R"(,
                                "stride_w": )" + strides + R"(,
                                "stride_h": )" + strides + R"(,
                                "fused_activation_function": )" + activation + R"(
                            },
                            "custom_options_format": "FLEXBUFFERS"
                        }
                    ],
                } ],
                "buffers" : [
                    { },
                    { },
                    { "data": )" + filterData + R"(, },
                    { "data": )" + biasData + R"(, },
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

struct SimpleConv3DWithBiasesFixture : Conv3DWithBiasesFixture
{
    SimpleConv3DWithBiasesFixture()
    : Conv3DWithBiasesFixture("[ 1, 2, 2, 2, 1 ]",      // inputShape
                              "[ 1, 2, 2, 2, 1 ]",      // outputShape
                              "[ 2, 2, 2, 1, 1 ]",      // filterShape
                              "[ 2,1, 1,0, 0,1, 1,1 ]", // filterData
                              "[ 1 ]",                  // biasShape
                              "[ 5, 0, 0, 0 ]",         // biasData
                              "1")                      // stride d, w and h
    {}
};

TEST_CASE_FIXTURE(SimpleConv3DWithBiasesFixture, "ParseConv3DWithBias")
{
    RunTest<5,
            armnn::DataType::QAsymmU8>(0,
                                       { 1, 2, 3, 4, 5, 6, 7, 8 },
                                       { 33, 21, 23, 13, 28, 25, 27, 21 });
}

TEST_CASE_FIXTURE(SimpleConv3DWithBiasesFixture, "ParseDynamicConv3DWithBias")
{
    RunTest<5,
            armnn::DataType::QAsymmU8,
            armnn::DataType::QAsymmU8>(0,
                                       { { "inputTensor", { 2, 4, 6, 8, 10, 12, 14, 16 } } },
                                       { { "outputTensor", {  61, 37, 41, 21, 51, 45, 49, 37 } } },
                                       true);
}

struct Relu6Conv3DWithBiasesFixture : Conv3DWithBiasesFixture
{
    Relu6Conv3DWithBiasesFixture()
    : Conv3DWithBiasesFixture("[ 1, 2, 2, 2, 1 ]",       // inputShape
                              "[ 1, 2, 2, 2, 1 ]",       // outputShape
                              "[ 2, 2, 2, 1, 1 ]",       // filterShape
                              "[ 2,1, 1,0, 0,1, 1,1 ]",  // filterData
                              "[ 1 ]",                   // biasShape
                              "[ 0, 0, 0, 0 ]",          // biasData
                              "1",                       // stride d, w, and h
                              "RELU6",                   // activation
                              "1.0",                     // filter scale
                              "0",                       // filter zero point
                              "2.0",                     // output scale
                              "0")                       // output zero point
    {}
};

TEST_CASE_FIXTURE(Relu6Conv3DWithBiasesFixture, "ParseConv3DAndRelu6WithBias")
{
    uint8_t relu6Min = 6 / 2; // Divide by output scale

    RunTest<5, armnn::DataType::QAsymmU8>(
        0,
        {
           1, 2, 3, 4, 5, 6, 7, 8
        },
        // RELU6 cuts output values at +6
        {
            std::min(relu6Min, static_cast<uint8_t>((1*2 + 2*1 + 3*1 + 4*0 + 5*0 + 6*1 + 7*1 + 8*1)/2)),
            std::min(relu6Min, static_cast<uint8_t>((2*2 + 0*1 + 0*1 + 0*0 + 0*0 + 0*1 + 8*1 + 0*1)/2)),
            std::min(relu6Min, static_cast<uint8_t>((3*2 + 0*1 + 0*1 + 0*0 + 0*0 + 8*1 + 0*1 + 0*1)/2)),
            std::min(relu6Min, static_cast<uint8_t>((4*2 + 0*1 + 0*1 + 0*0 + 8*0 + 0*1 + 0*1 + 0*1)/2)),
            std::min(relu6Min, static_cast<uint8_t>((5*2 + 0*1 + 0*1 + 8*0 + 0*0 + 0*1 + 0*1 + 0*1)/2)),
            std::min(relu6Min, static_cast<uint8_t>((6*2 + 0*1 + 8*1 + 0*0 + 0*0 + 0*1 + 0*1 + 0*1)/2)),
            std::min(relu6Min, static_cast<uint8_t>((7*2 + 8*1 + 0*1 + 0*0 + 0*0 + 0*1 + 0*1 + 0*1)/2)),
            std::min(relu6Min, static_cast<uint8_t>((8*2 + 0*1 + 0*1 + 0*0 + 0*0 + 0*1 + 0*1 + 0*1)/2))
        });
}

}
#endif
