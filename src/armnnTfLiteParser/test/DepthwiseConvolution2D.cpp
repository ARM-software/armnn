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

struct DepthwiseConvolution2dFixture2 : public ParserFlatbuffersFixture
{
    explicit DepthwiseConvolution2dFixture2(const std::string& inputShape,
                                            const std::string& outputShape,
                                            const std::string& filterShape,
                                            const std::string& filterData,
                                            const std::string& strides,
                                            const std::string& paddingType,
                                            const std::string  biasShape                = "",
                                            const std::string  biasData                 = "",
                                            const std::string  filter_quant_min         = "[ 0.0 ]",
                                            const std::string  filter_quant_max         = "[ 255.0 ]",
                                            const std::string  filter_quant_scale       = "[ 1.0 ]",
                                            const std::string  filter_quant_zero_point  = "[ 0 ]",
                                            const std::string  filter_quant_axis        = ""
                                            )
    {
        std::string inputTensors = "[ 0, 2 ]";
        std::string biasTensor   = "";
        std::string biasBuffer   = "";
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

        std::string filter_qantization =
                               R"(
                                "min": )" + filter_quant_min + R"(,
                                "max": )" + filter_quant_max + R"(,
                                "scale": )" + filter_quant_scale + R"(,
                                "zero_point": )" + filter_quant_zero_point;
        // A given quantization axis indicates if per channel quantization is used for filters
        if (filter_quant_axis.size() > 0)
        {
            filter_qantization +=
                               R"(,
                                "quantized_dimension": )" + filter_quant_axis;
        }
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "DEPTHWISE_CONV_2D" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": )" + inputShape + R"(,
                            "type": "INT8",
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
                            "type": "INT8",
                            "buffer": 1,
                            "name": "outputTensor",
                            "quantization": {
                                "min": [ 0.0 ],
                                "max": [ 511.0 ],
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + filterShape + R"(,
                            "type": "INT8",
                            "buffer": 2,
                            "name": "filterTensor",
                            "quantization": {)" + filter_qantization + R"(
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


// No quantization meaning scale=1.0 and offset=0.0 and tensor quantization
struct DepthwiseConvolution2dNoQuantFixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dNoQuantFixture()
    : DepthwiseConvolution2dFixture2("[ 1, 3, 3, 3 ]",           // inputShape
                                     "[ 1, 3, 3, 3 ]",           // outputShape
                                     "[ 1, 3, 3, 3 ]",           // filterShape
                                     "[ 9,8,7, 6,5,4, 3,2,1, "
                                       "9,8,7, 6,5,4, 3,2,1, "
                                       "9,8,7, 6,5,4, 3,2,1 ]",  // filterData
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     ""                          // bias data
                                    )
    {}
};

// No quantization meaning scale=1.0 and offset=0.0 and tensor quantization
BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DNoQuant, DepthwiseConvolution2dNoQuantFixture)
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        { 18, 14, 10, 36, 30, 24, 30, 26, 22, 27, 21, 15, 54, 45,
          36, 45, 39, 33, 18, 14, 10, 36, 30, 24, 30, 26, 22});
}

// Uses per channel quantization on weights but with scales = 1.0 and offsets = 0.0
struct DepthwiseConvolution2dNoChannelQuantFixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dNoChannelQuantFixture()
    : DepthwiseConvolution2dFixture2("[ 1, 3, 3, 3 ]",           // inputShape
                                     "[ 1, 3, 3, 3 ]",           // outputShape
                                     "[ 1, 3, 3, 3 ]",           // filterShape
                                     "[ 9,8,7, 6,5,4, 3,2,1, 9,8,7, 6,5,4, 3,2,1, 9,8,7, 6,5,4, 3,2,1 ]",  // filterData
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     "",                         // bias data
                                     "[ 0.0 ]",                  // filter quantization min values
                                     "[ 255.0 ]",                // filter quantization max values
                                     "[ 1.0, 1.0, 1.0]",         // filter quantization scales
                                     "[ 0, 0, 0]",               // filter quantization zero-points
                                     "3"                         // filter quantized axis
                                                                 // (in case of per channel quantization)
                                    )
    {}
};

// Uses per channel quantization on weights but with scales = 1.0 and offsets = 0.0
BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DFilterNoChannelQuant, DepthwiseConvolution2dNoChannelQuantFixture)
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        { 18, 14, 10, 36, 30, 24, 30, 26, 22, 27, 21, 15, 54, 45,
          36, 45, 39, 33, 18, 14, 10, 36, 30, 24, 30, 26, 22});
}

// Uses per channel quantization on weights but all scales are set to the same value
struct DepthwiseConvolution2dWeightsPerChannelQuantFixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuantFixture()
    : DepthwiseConvolution2dFixture2("[ 1, 3, 3, 3 ]",           // inputShape
                                     "[ 1, 3, 3, 3 ]",           // outputShape
                                     "[ 1, 3, 3, 3 ]",           // filterShape
                                     // filterData is [ 9,8,7, 6,5,4, 3,2,1, 9,8,7, 6,5,4, 3,2,1, 9,8,7, 6,5,4, 3,2,1 ]
                                     // quantized per channel with q_dim=3
                                     "[36, 32, 28, 24, 20, 16, 12,  8,  4, 36, 32, 28, 24, "
                                      "20, 16, 12,  8,  4, 36, 32, 28, 24, 20, 16, 12,  8, 4]",
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     "",                         // bias data
                                     "[ 0.0 ]",                  // filter quantization min values
                                     "[ 255.0 ]",                // filter quantization max values
                                     "[ 0.25, 0.25, 0.25]",      // filter quantization scales
                                     "[ 0, 0, 0]",               // filter quantization zero-points
                                     "3"                         // filter quantized axis
                                                                 // (in case of per channel quantization)
                                    )
    {}
};

// Weights are per channel quantized but all scales are set to the same value
BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DFilterWeightsPerChannelQuant,
                        DepthwiseConvolution2dWeightsPerChannelQuantFixture)
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        { 18, 14, 10, 36, 30, 24, 30, 26, 22, 27, 21, 15, 54, 45,
          36, 45, 39, 33, 18, 14, 10, 36, 30, 24, 30, 26, 22});
}

// Uses per channel quantization on weights all scales are different in this test
struct DepthwiseConvolution2dWeightsPerChannelQuant1Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant1Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 3, 3, 3 ]",           // inputShape
                                     "[ 1, 3, 3, 3 ]",           // outputShape
                                     "[ 1, 3, 3, 3 ]",           // filterShape
                                     // filterData is [ 9,8,7, 6,5,4, 3,2,1, 9,8,7, 6,5,4, 3,2,1, 9,8,7, 6,5,4, 3,2,1 ]
                                     // quantized per channel with q_dim=3
                                     "[36, 40, 70, 24, 25, 40, 12, 10, 10, 36, 40, 70, 24, "
                                      "25, 40, 12, 10, 10, 36, 40, 70, 24, 25, 40, 12, 10, 10]",
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     "",                         // bias data
                                     "[ 0.0 ]",                  // filter quantization min values
                                     "[ 255.0 ]",                // filter quantization max values
                                     "[ 0.25, 0.2, 0.1]",        // filter quantization scales
                                     "[ 0, 0, 0]",               // filter quantization zero-points
                                     "3"                         // filter quantized axis
                                                                 // (in case of per channel quantization)
                                    )
    {}
};

// Uses per channel quantization on weights all scales are different in this test
BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DFilterWeightsPerChannelQuant1,
                        DepthwiseConvolution2dWeightsPerChannelQuant1Fixture)
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        { 18, 14, 10, 36, 30, 24, 30, 26, 22, 27, 21, 15, 54, 45,
          36, 45, 39, 33, 18, 14, 10, 36, 30, 24, 30, 26, 22});
}


// Uses per channel quantization on weights all scales are different in this test
// Uses different shape for weights and input compared to the other tests above
struct DepthwiseConvolution2dWeightsPerChannelQuant2Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant2Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 4, 4, 4 ]",           // inputShape
                                     "[ 1, 4, 4, 4 ]",           // outputShape
                                     "[ 1, 2, 2, 4 ]",           // filterShape
                                     // filterData is [ 9,8,7,6, 5,4,3,2, 1,9,8,7, 6,5,4,3 ]
                                     // quantized per channel with q_dim=3
                                     "[36, 40, 70, 20, 20, 20, 30, 6, 4, 45, 80, 23, 24, 25, 40, 10]",
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     "",                         // bias data
                                     "[ 0.0 ]",                  // filter quantization min values
                                     "[ 255.0 ]",                // filter quantization max values
                                     "[ 0.25, 0.2, 0.1, 0.3]",   // filter quantization scales
                                     "[ 0, 0, 0, 0]",            // filter quantization zero-points
                                     "3"                         // filter quantized axis
                                                                 // (in case of per channel quantization)
                                    )
    {}
};

// Uses per channel quantization on weights all scales are different in this test
// Uses different shape for weights and input compared to the other tests above
BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DFilterWeightsPerChannelQuant2,
                        DepthwiseConvolution2dWeightsPerChannelQuant2Fixture)
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
          1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
          1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
          1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1},
        { 21, 26, 22, 18, 21, 26, 22, 18, 21, 26, 22, 18, 10, 17, 15, 13,
          21, 26, 22, 18, 21, 26, 22, 18, 21, 26, 22, 18, 10, 17, 15, 13,
          21, 26, 22, 18, 21, 26, 22, 18, 21, 26, 22, 18, 10, 17, 15, 13,
          14, 12, 10,  8, 14, 12, 10,  8, 14, 12, 10,  8,  9,  8,  7,  6});
}

// Test for depthwise_multiplier different to one (M > 1)
struct DepthwiseConvolution2dWeightsPerChannelQuant4Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant4Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 4, 4, 4 ]",            // inputShape
                                     "[ 1, 4, 4, 16 ]",           // outputShape
                                     "[ 1, 2, 2, 16 ]",           // filterShape
                                     // filter data is [ 9,8,7,6, 5,4,3,2, 1,9,8,7, 6,5,4,3,
                                     //                  9,8,7,6, 5,4,3,2, 1,9,8,7, 6,5,4,3,
                                     //                  9,8,7,6, 5,4,3,2, 1,9,8,7, 6,5,4,3,
                                     //                  9,8,7,6, 5,4,3,2, 1,9,8,7, 6,5,4,3 ]
                                     //                  quantized per channel with q_dim=3
                                     "[36, 40, 70, 20, 20, 20, 30, 6, 4, 45, 80, 23, 24, 25, 40, 10, "
                                      "36, 40, 70, 20, 20, 20, 30, 6, 4, 45, 80, 23, 24, 25, 40, 10, "
                                      "36, 40, 70, 20, 20, 20, 30, 6, 4, 45, 80, 23, 24, 25, 40, 10, "
                                      "36, 40, 70, 20, 20, 20, 30, 6, 4, 45, 80, 23, 24, 25, 40, 10]",
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     "",                         // bias data
                                     "[ 0.0 ]",                  // filter quantization min values
                                     "[ 255.0 ]",                // filter quantization max values
                                     "[ 0.25, 0.2, 0.1, 0.3,"
                                       "0.25, 0.2, 0.1, 0.3,"
                                       "0.25, 0.2, 0.1, 0.3,"
                                       "0.25, 0.2, 0.1, 0.3]",   // filter quantization scales
                                     "[ 0, 0, 0, 0]",            // filter quantization zero-points
                                     "3"                         // filter quantized axis
                                                                 // (in case of per channel quantization)
                                    )
    {}
};

// Test for depthwise_multiplier different to one (M > 1)
BOOST_FIXTURE_TEST_CASE(ParseDepthwiseConv2DFilterWeightsPerChannelQuant4,
                        DepthwiseConvolution2dWeightsPerChannelQuant4Fixture)
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
          1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
          1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
          1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1},
        { 36, 32, 28, 24, 20, 16, 12,  8, 4, 36, 32, 28, 24, 20, 16, 12,
          36, 32, 28, 24, 20, 16, 12,  8, 4, 36, 32, 28, 24, 20, 16, 12,
          36, 32, 28, 24, 20, 16, 12,  8, 4, 36, 32, 28, 24, 20, 16, 12,
          18, 16, 14, 12, 10,  8,  6,  4, 2, 18, 16, 14, 12, 10,  8,  6,
          36, 32, 28, 24, 20, 16, 12,  8, 4, 36, 32, 28, 24, 20, 16, 12,
          36, 32, 28, 24, 20, 16, 12,  8, 4, 36, 32, 28, 24, 20, 16, 12,
          36, 32, 28, 24, 20, 16, 12,  8, 4, 36, 32, 28, 24, 20, 16, 12,
          18, 16, 14, 12, 10,  8,  6,  4, 2, 18, 16, 14, 12, 10,  8,  6,
          36, 32, 28, 24, 20, 16, 12,  8, 4, 36, 32, 28, 24, 20, 16, 12,
          36, 32, 28, 24, 20, 16, 12,  8, 4, 36, 32, 28, 24, 20, 16, 12,
          36, 32, 28, 24, 20, 16, 12,  8, 4, 36, 32, 28, 24, 20, 16, 12,
          18, 16, 14, 12, 10,  8,  6,  4, 2, 18, 16, 14, 12, 10,  8,  6,
          18, 16, 14, 12, 10,  8,  6,  4, 2, 18, 16, 14, 12, 10,  8,  6,
          18, 16, 14, 12, 10,  8,  6,  4, 2, 18, 16, 14, 12, 10,  8,  6,
          18, 16, 14, 12, 10,  8,  6,  4, 2, 18, 16, 14, 12, 10,  8,  6,
           9,  8,  7,  6,  5,  4,  3,  2, 1,  9,  8,  7,  6,  5,  4,  3});
}

BOOST_AUTO_TEST_SUITE_END()
