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

struct DepthwiseConvolution2dFixture : public ParserFlatbuffersFixture
{
    explicit DepthwiseConvolution2dFixture(const std::string& inputShape,
                                           const std::string& outputShape,
                                           const std::string& filterShape,
                                           const std::string& filterData,
                                           const std::string& strides,
                                           const std::string& paddingType,
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
                "operator_codes": [ { "builtin_code": "DEPTHWISE_CONV_2D" } ],
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
                            "builtin_options_type": "DepthwiseConv2DOptions",
                            "builtin_options": {
                                "padding": ")" + paddingType + R"(",
                                "stride_w": )" + strides+ R"(,
                                "stride_h": )" + strides+ R"(,
                                "depth_multiplier": 1,
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

struct DepthwiseConvolution2dSameFixture : DepthwiseConvolution2dFixture
{
    DepthwiseConvolution2dSameFixture()
    : DepthwiseConvolution2dFixture("[ 1, 3, 3, 1 ]",           // inputShape
                                    "[ 1, 3, 3, 1 ]",           // outputShape
                                    "[ 1, 3, 3, 1 ]",           // filterShape
                                    "[ 9,8,7, 6,5,4, 3,2,1 ]",  // filterData
                                    "1",                        // stride w and h
                                    "SAME")                     // padding type
    {}
};

BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DSame, DepthwiseConvolution2dSameFixture)
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        { 0, 1, 2,
          3, 4, 5,
          6, 7, 8 },
        // the expected values were generated using the example python implementation at
        // https://eli.thegreenplace.net/2018/depthwise-separable-convolutions-for-machine-learning/
        // divide the expected values by the output scale, as it is not 1.0
        {  14/2,  35/2,  38/2,
           57/2, 120/2, 111/2,
          110/2, 197/2, 158/2 });
}

struct DepthwiseConvolution2dValidFixture : DepthwiseConvolution2dFixture
{
    DepthwiseConvolution2dValidFixture ()
    : DepthwiseConvolution2dFixture("[ 1, 3, 3, 1 ]",           // inputShape
                                    "[ 1, 1, 1, 1 ]",           // outputShape
                                    "[ 1, 3, 3, 1 ]",           // filterShape
                                    "[ 9,8,7, 6,5,4, 3,2,1 ]",  // filterData
                                    "1",                        // stride w and h
                                    "VALID")                    // padding type
    {}
};

BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DValid, DepthwiseConvolution2dValidFixture)
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        { 0, 1, 2,
          3, 4, 5,
          6, 7, 8 },
        // divide the expected values by the output scale, as it is not 1.0
        { 120/2 });
}

struct DepthwiseConvolution2dSameBiasFixture : DepthwiseConvolution2dFixture
{
    DepthwiseConvolution2dSameBiasFixture()
    : DepthwiseConvolution2dFixture("[ 1, 3, 3, 1 ]",           // inputShape
                                    "[ 1, 3, 3, 1 ]",           // outputShape
                                    "[ 1, 3, 3, 1 ]",           // filterShape
                                    "[ 9,8,7, 6,5,4, 3,2,1 ]",  // filterData
                                    "1",                        // stride w and h
                                    "SAME",                     // padding type
                                    "[ 1 ]",                    // biasShape
                                    "[ 10, 0, 0, 0 ]")          // biasData
    {}
};

BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DSameBias, DepthwiseConvolution2dSameBiasFixture)
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        { 0, 1, 2,
          3, 4, 5,
          6, 7, 8 },
        // divide the expected values by the output scale, as it is not 1.0
        { ( 14+10)/2, ( 35+10)/2, ( 38+10)/2,
          ( 57+10)/2, (120+10)/2, (111+10)/2,
          (110+10)/2, (197+10)/2, (158+10)/2 });
}

struct DynamicDepthwiseConvolution2dSameBiasFixture : DepthwiseConvolution2dFixture
{
    DynamicDepthwiseConvolution2dSameBiasFixture()
        : DepthwiseConvolution2dFixture("[ 1, 3, 3, 1 ]",           // inputShape
                                        "[ ]",           // outputShape
                                        "[ 1, 3, 3, 1 ]",           // filterShape
                                        "[ 9,8,7, 6,5,4, 3,2,1 ]",  // filterData
                                        "1",                        // stride w and h
                                        "SAME",                     // padding type
                                        "[ 1 ]",                    // biasShape
                                        "[ 10, 0, 0, 0 ]")          // biasData
    {}
};

BOOST_FIXTURE_TEST_CASE(ParseDynamicDepthwiseConv2DSameBias, DynamicDepthwiseConvolution2dSameBiasFixture)
{
    RunTest<4, armnn::DataType::QAsymmU8, armnn::DataType::QAsymmU8>(0,
                                                      { { "inputTensor", { 0, 1, 2,
                                                                            3, 4, 5,
                                                                            6, 7, 8 } } },
                                                      { { "outputTensor", { ( 14+10)/2, ( 35+10)/2, ( 38+10)/2,
                                                                            ( 57+10)/2, (120+10)/2, (111+10)/2,
                                                                            (110+10)/2, (197+10)/2, (158+10)/2  } } },
                                                      true);
}

BOOST_AUTO_TEST_SUITE_END()
