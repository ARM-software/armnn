//
// Copyright © 2017 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_DepthwiseConvolution2D")
{
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

TEST_CASE_FIXTURE(DepthwiseConvolution2dSameFixture, "ParseDepthwiseConv2DSame")
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

TEST_CASE_FIXTURE(DepthwiseConvolution2dValidFixture, "ParseDepthwiseConv2DValid")
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

TEST_CASE_FIXTURE(DepthwiseConvolution2dSameBiasFixture, "ParseDepthwiseConv2DSameBias")
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

TEST_CASE_FIXTURE(DynamicDepthwiseConvolution2dSameBiasFixture, "ParseDynamicDepthwiseConv2DSameBias")
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
                                           const std::string biasShape = "",
                                           const std::string biasData = "",
                                           const std::string filter_quant_min = "[ 0.0 ]",
                                           const std::string filter_quant_max = "[ 255.0 ]",
                                           const std::string filter_quant_scale = "[ 1.0 ]",
                                           const std::string filter_quant_zero_point = "[ 0 ]",
                                           const std::string filter_quant_axis = "",
                                           const std::string output_scale = "[ 1.0 ]")
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
                                "scale": )" + output_scale + R"(,
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
TEST_CASE_FIXTURE(DepthwiseConvolution2dNoQuantFixture, "ParseDepthwiseConv2DNoQuant")
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
                                     "[ 9,8,7, 6,5,4, 3,2,1, 9,8,7, 6,5,4, 3,2,1, 9,8,7, 6,5,4, 3,2,1 ]",  //filterData
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
TEST_CASE_FIXTURE(DepthwiseConvolution2dNoChannelQuantFixture, "ParseDepthwiseConv2DFilterNoChannelQuant")
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
TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuantFixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant")
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
TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant1Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant1")
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
TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant2Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant2")
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
TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant4Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant4")
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


struct DepthwiseConvolution2dWeightsPerChannelQuant6Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant6Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 4, 4, 4 ]",            // inputShape
                                     "[ 1, 4, 4, 16 ]",           // outputShape
                                     "[ 1, 2, 2, 16 ]",           // filterShape
                                     // filter data is [ 3,4,1,1,1,3,3,2,1,4,3,4,1,2,2,4,
                                     //                  2,0,3,1,0,2,4,3,4,3,0,1,3,4,4,1,
                                     //                  3,3,2,0,0,0,1,3,3,2,4,4,3,1,1,3,
                                     //                  1,0,0,2,3,0,1,1,4,2,2,1,2,3,2,0]
                                     //                  quantized per channel with q_dim=3
                                     "[12,20,10, 3, 4,15,30, 6, 4,20,30,12, 4,10,20,12,"
                                       " 8, 0,30, 3, 0,10,40, 9,16,15, 0, 3,12,20,40, 3,"
                                       " 12,15,20, 0, 0, 0,10, 9,12,10,40,12,12, 5,10, 9,"
                                       " 4, 0, 0, 6,12, 0,10, 3,16,10,20, 3, 8,15,20, 0]",
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     "",                         // bias data
                                     "[ 0.0 ]",                  // filter quantization min values
                                     "[ 255.0 ]",                // filter quantization max values
                                     "[ 0.25, 0.2, 0.1, 0.333333333,"
                                       "0.25, 0.2, 0.1, 0.333333333,"
                                       "0.25, 0.2, 0.1, 0.333333333,"
                                       "0.25, 0.2, 0.1, 0.333333333]",   // filter quantization scales
                                     "[ 0, 0, 0, 0]",            // filter quantization zero-points
                                     "3"                         // filter quantized axis
                                                                 // (in case of per channel quantization)
                                    )
    {}
};


TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant6Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant6")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1,0,1,2,0,4,4,0,2,1,2,0,1,3,3,0,
          1,2,2,3,3,4,1,1,2,4,1,3,4,2,0,2,
          0,3,1,3,4,3,2,0,1,2,3,3,0,2,4,2,
          1,2,1,4,3,4,1,3,1,0,2,3,1,3,2,0},
        {  9, 7, 3, 7,12, 8,22,22,27,22,13,17,13,10, 9,17,
          15, 9,12, 6,16,14,24,27,19,26,18,23, 9,10, 7, 3,
          18,14, 9,11, 7, 9,21,25,17,19,10,15,13, 9, 7, 9,
          15,16, 9, 1, 3, 9,11,12, 3,12, 9,12, 6, 2, 2, 6,
          13, 4,10,12,11,14,28,28,17,17,14,15,15,13,13,22,
          26,24,17, 7,10,20,33,31,23,17,17,16,16,23,20, 7,
          17,11,16, 6,10,16,24,22,26,18,23,20,22,23,21,23,
          12,16, 4, 4, 2, 6, 8,10,12, 8,16,16, 8, 6, 6,14,
          14, 3,14,10,15,15,27,25,16,14, 9,11,21,19,16,24,
          24,25,13, 7, 3,13,21,24,25,23,14,17,24,24,21,12,
           7, 7, 3, 3,11,10,17,13,33,32,21,26,18,17,17,23,
           3, 3, 2, 0, 2, 6, 9,13,10,20,20,24, 2, 4, 4, 8,
           9, 4,10, 4, 2,14,22,16, 5, 7, 3, 5,13,20,20,19,
          11,12, 6, 4, 4,12,12, 8, 9,10, 3, 6,12,18,18,15,
           5, 4, 4, 2, 0, 6,12, 9,10,14, 6,10, 3, 6, 6,12,
           3, 4, 1, 1, 3, 9, 9, 6, 2, 8, 6, 8, 0, 0, 0, 0});
}


struct DepthwiseConvolution2dWeightsPerChannelQuant1_1Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant1_1Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 3, 3, 3 ]",           // inputShape
                                     "[ 1, 3, 3, 3 ]",           // outputShape
                                     "[ 1, 3, 3, 3 ]",           // filterShape
                                     // filterData is [ 1,4,0,2,4,3,1,0,1,
                                     //                 3,0,4,0,1,3,4,2,4,
                                     //                 3,0,3,4,4,0,3,4,2]
                                     // quantized per channel with q_dim=3
                                     "[ 4,20, 0, 8,20,30, 4, 0,10,12,"
                                     " 0,40, 0, 5,30,16,10,40,12, 0,"
                                       "30,16,20, 0,12,20,20]",
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


TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant1_1Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant1_1")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
        { 11,11, 9,17,11,16,10, 5,10,
          14,15,13,21,19,20,13,13,13,
          7, 7,11,11,11,15, 6, 9,10});
}

// Same with input different to 1
struct DepthwiseConvolution2dWeightsPerChannelQuant1_2Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant1_2Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 3, 3, 3 ]",           // inputShape
                                     "[ 1, 3, 3, 3 ]",           // outputShape
                                     "[ 1, 3, 3, 3 ]",           // filterShape
                                     // filterData is [ 1,4,0,2,4,3,1,0,1,
                                     //                 3,0,4,0,1,3,4,2,4,
                                     //                 3,0,3,4,4,0,3,4,2]
                                     // quantized per channel with q_dim=3
                                     "[ 4,20, 0, 8,20,30, 4, 0,10,12,"
                                     " 0,40, 0, 5,30,16,10,40,12, 0,"
                                       "30,16,20, 0,12,20,20]",
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


TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant1_2Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant1_2")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 3,2,0,0,4,3,0,1,2,
          0,1,3,0,4,2,2,2,3,
          2,4,3,2,0,4,3,4,0},
        {  0,30,16,15,30,32, 8, 9,24,
           20,33,28,34,48,50,18,38,35,
           8, 8,36,20,28,33,10,28,25});
}


struct DepthwiseConvolution2dWeightsPerChannelQuant4_1Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant4_1Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 4, 4, 4 ]",            // inputShape
                                     "[ 1, 4, 4, 16 ]",           // outputShape
                                     "[ 1, 2, 2, 16 ]",           // filterShape
                                     // filter data is [ 3,4,1,1,1,3,3,2,1,4,3,4,1,2,2,4,
                                     //                  2,0,3,1,0,2,4,3,4,3,0,1,3,4,4,1,
                                     //                  3,3,2,0,0,0,1,3,3,2,4,4,3,1,1,3,
                                     //                  1,0,0,2,3,0,1,1,4,2,2,1,2,3,2,0 ]
                                     //                  quantized per channel with q_dim=3
                                     "[12,20,10, 3, 4,15,30, 6, 4,20,30,13, 4,10,20,13,"
                                     "  8, 0,30, 3, 0,10,40,10,16,15, 0, 3,12,20,40, 3,"
                                     " 12,15,20, 0, 0, 0,10,10,12,10,40,13,12, 5,10,10,"
                                     "  4, 0, 0, 6,12, 0,10, 3,16,10,20, 3, 8,15,20, 0]",
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


TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant4_1Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant4_1")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
          1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
          1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1,
          1,1,1,1, 1,1,1,1, 1,1,1,1, 1,1,1,1},
        {  9, 7, 6, 4, 4, 5, 9, 9,12,11, 9,10, 9,10, 9, 8,
           9, 7, 6, 4, 4, 5, 9, 9,12,11, 9,10, 9,10, 9, 8,
           9, 7, 6, 4, 4, 5, 9, 9,12,11, 9,10, 9,10, 9, 8,
           6, 7, 3, 1, 1, 3, 4, 5, 4, 6, 7, 8, 4, 3, 3, 7,
           9, 7, 6, 4, 4, 5, 9, 9,12,11, 9,10, 9,10, 9, 8,
           9, 7, 6, 4, 4, 5, 9, 9,12,11, 9,10, 9,10, 9, 8,
           9, 7, 6, 4, 4, 5, 9, 9,12,11, 9,10, 9,10, 9, 8,
           6, 7, 3, 1, 1, 3, 4, 5, 4, 6, 7, 8, 4, 3, 3, 7,
           9, 7, 6, 4, 4, 5, 9, 9,12,11, 9,10, 9,10, 9, 8,
           9, 7, 6, 4, 4, 5, 9, 9,12,11, 9,10, 9,10, 9, 8,
           9, 7, 6, 4, 4, 5, 9, 9,12,11, 9,10, 9,10, 9, 8,
           6, 7, 3, 1, 1, 3, 4, 5, 4, 6, 7, 8, 4, 3, 3, 7,
           5, 4, 4, 2, 1, 5, 7, 5, 5, 7, 3, 5, 4, 6, 6, 5,
           5, 4, 4, 2, 1, 5, 7, 5, 5, 7, 3, 5, 4, 6, 6, 5,
           5, 4, 4, 2, 1, 5, 7, 5, 5, 7, 3, 5, 4, 6, 6, 5,
           3, 4, 1, 1, 1, 3, 3, 2, 1, 4, 3, 4, 1, 2, 2, 4});
}



struct DepthwiseConvolution2dWeightsPerChannelQuant4_2Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant4_2Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 4, 4, 4 ]",            // inputShape
                                     "[ 1, 4, 4, 16 ]",           // outputShape
                                     "[ 1, 2, 2, 16 ]",           // filterShape
                                     // filter data is [ 3,4,1,1,1,3,3,2,1,4,3,4,1,2,2,4,
                                     //                  2,0,3,1,0,2,4,3,4,3,0,1,3,4,4,1,
                                     //                  3,3,2,0,0,0,1,3,3,2,4,4,3,1,1,3,
                                     //                  1,0,0,2,3,0,1,1,4,2,2,1,2,3,2,0 ]
                                     //                  quantized per channel with q_dim=3
                                     "[12,20,10, 3, 4,15,30, 6, 4,20,30,13, 4,10,20,13,"
                                     "  8, 0,30, 3, 0,10,40,10,16,15, 0, 3,12,20,40, 3,"
                                     " 12,15,20, 0, 0, 0,10,10,12,10,40,13,12, 5,10,10,"
                                     "  4, 0, 0, 6,12, 0,10, 3,16,10,20, 3, 8,15,20, 0]",
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


TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant4_2Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant4_2")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 3,3,3,4, 4,4,0,0, 0,3,4,3, 0,2,2,3,
          3,0,3,0, 0,3,2,1, 4,1,2,2, 0,0,0,4,
          3,2,2,2, 2,1,0,4, 4,3,2,4, 3,2,0,0,
          4,1,4,4, 1,0,4,3, 3,2,0,3, 1,1,0,2},
        { 26,21,21, 7,12,17,28,21,20,22,25,26, 6,11,10,16,
          16,16, 4,12, 7,18,28,27,30,20,12,14,16,19,17, 6,
          12,12, 8, 0, 3,13,18,15,18,26,20,26,26,32,28,21,
          0, 0, 0, 0, 2, 6, 6, 4, 2, 8, 6, 8,15,10,10,24,
          20,21, 9, 7, 3, 6,15,16,17,22,17,22,17,18,14, 7,
          18, 6,16,12,12,11,17,15,18,18,10,12,27,26,22,18,
          27,28,12,10, 7, 3, 8,13, 8,12,14,16,26,24,24,24,
          9, 9, 6, 0, 0, 0, 2, 6, 0, 0, 0, 0, 4, 8, 8,16,
          26,24,17, 7, 2, 8,11,10,30,24,30,28,32,33,30,24,
          20,11,16,12, 7, 9,17,13,20,14,16,18,31,36,33,29,
          28,25,19, 9, 6,13,20,19, 2, 8, 6, 8,17,17,15,25,
          12,15, 5, 3, 2, 6, 7, 7, 0, 0, 0, 0, 6, 2, 2, 6,
          14,16, 7, 5, 1, 3, 3, 2,20,28,12,20,13,20,20,19,
          9, 4,10, 4, 0, 4, 8, 6, 4,16,12,16,12,18,18,15,
          11,12, 6, 4, 2, 8,10, 7, 0, 0, 0, 0, 9,14,14,14,
          3, 4, 1, 1, 1, 3, 3, 2, 0, 0, 0, 0, 2, 4, 4, 8});
}


struct DepthwiseConvolution2dWeightsPerChannelQuant4_5Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant4_5Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 4, 4, 4 ]",            // inputShape
                                     "[ 1, 4, 4, 16 ]",           // outputShape
                                     "[ 1, 2, 2, 16 ]",           // filterShape
                                     // filter data is [     1,   4,   9,  16,  25,  36,
                                     //                     49,  64,  81, 100, 121, 144,
                                     //                    169, 196, 225, 256,  17,  36,
                                     //                     57,  80, 105, 132, 161, 192,
                                     //                    225, 260, 297, 336, 377, 420,
                                     //                    465, 512,  33,  68, 105, 144,
                                     //                    185, 228, 273, 320, 369, 420,
                                     //                    473, 528, 585, 644, 705, 768,
                                     //                     49, 100, 153, 208, 265, 324,
                                     //                    385, 448, 513, 580, 649, 720,
                                     //                    793, 868, 945,1024 ]
                                     //                  quantized per channel with q_dim=3
                                     "[ 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,"
                                       " 17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,"
                                       " 33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,"
                                       "49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64]",
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     "",                         // bias data
                                     "[ 0.0 ]",                  // filter quantization min values
                                     "[ 255.0 ]",                // filter quantization max values
                                     "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,11,12,13,14,15,16]", // filter quantization scales
                                     "[ 0, 0, 0, 0]",            // filter quantization zero-points
                                     "3",                         // filter quantized axis
                                                                 // (in case of per channel quantization)
                                     "[ 100.0 ]"                  // output scale
                                    )
    {}
};

// Test for depthwise_multiplier different to one (M > 1)
TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant4_5Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant4_5")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 1,1,1,2,2,2,1,2,1,2,2,1,2,2,1,1,1,1,1,1,1,2,2,2,
          1,2,2,2,1,1,1,2,1,1,1,1,2,1,2,1,2,1,1,2,1,2,1,1,
          1,2,2,1,2,2,1,1,2,1,2,1,1,2,1,2},
        {  1, 2, 3, 5, 9,11,14,16,17,19,21,24,32,36,39,43,
           1, 2, 3, 4,11,14,17,20,22,26,29,33,34,38,42,46,
           1, 2, 3, 5, 8,11,13,16,16,18,21,24,33,36,39,43,
           0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6,13,14,16,17,
           1, 3, 4, 6, 6, 8,10,12,19,22,24,27,23,25,28,30,
           1, 3, 5, 8, 7, 8,10,12,18,21,24,27,32,36,39,43,
           1, 2, 4, 5, 8,10,13,15,12,14,16,18,30,33,37,40,
           0, 0, 1, 1, 3, 4, 5, 7, 4, 5, 5, 6, 9,10,11,12,
           1, 3, 5, 7,10,12,15,17,17,20,23,25,19,21,23,25,
           2, 4, 6, 8, 7, 9,11,13,17,20,23,25,23,25,28,30,
           1, 2, 4, 6, 9,11,14,16,15,17,20,22,28,31,35,38,
           0, 0, 1, 1, 4, 5, 6, 7, 4, 5, 5, 6,13,14,16,17,
           0, 0, 1, 1, 2, 3, 4, 5, 3, 4, 5, 6, 5, 6, 6, 7,
           0, 0, 1, 1, 1, 2, 2, 3, 5, 6, 7, 8, 5, 6, 6, 7,
           0, 0, 0, 1, 2, 3, 3, 4, 3, 4, 5, 6, 9,10,11,12,
           0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 3, 3, 4, 5});
}


struct DepthwiseConvolution2dWeightsPerChannelQuant4_3_1Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant4_3_1Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 4, 4, 4 ]",            // inputShape
                                     "[ 1, 4, 4, 16 ]",           // outputShape
                                     "[ 1, 2, 2, 16 ]",           // filterShape
                                     // filter data is [ 3,4,1,1,1,3,3,2,1,4,3,4,1,2,2,4,
                                     //                  2,0,3,1,0,2,4,3,4,3,0,1,3,4,4,1,
                                     //                  3,3,2,0,0,0,1,3,3,2,4,4,3,1,1,3,
                                     //                  1,0,0,2,3,0,1,1,4,2,2,1,2,3,2,0 ]
                                     //                  quantized per channel with q_dim=3
                                     "[12,20,10, 3, 2,24, 9,10, 5,16,30,12, 3,10, 4,32,"
                                     "  8, 0,30, 3, 0,16,12,15,20,12, 0, 3, 9,20, 8, 8,"
                                     " 12,15,20, 0, 0, 0, 3,15,15, 8,40,12, 9, 5, 2,24,"
                                     "  4, 0, 0, 6, 6, 0, 3, 5,20, 8,20, 3, 6,15, 4, 0]",
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     "",                         // bias data
                                     "[ 0.0 ]",                  // filter quantization min values
                                     "[ 255.0 ]",                // filter quantization max values
                                     "[0.25, 0.2, 0.1, 0.3333333333, "
                                        "0.5, 0.125, 0.33333333, 0.2, "
                                        "0.2, 0.25, 0.1, 0.333333333, "
                                        "0.3333333333, 0.2, 0.5, 0.125]",   // filter quantization scales
                                     "[ 0, 0, 0, 0]",            // filter quantization zero-points
                                     "3"                         // filter quantized axis
                                                                 // (in case of per channel quantization)
                                    )
    {}
};

// Test for depthwise_multiplier different to one (M > 1)
TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant4_3_1Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant4_3_1")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 3,3,3,4, 4,4,0,0, 0,3,4,3, 0,2,2,3,
          3,0,3,0, 0,3,2,1, 4,1,2,2, 0,0,0,4,
          3,2,2,2, 2,1,0,4, 4,3,2,4, 3,2,0,0,
          4,1,4,4, 1,0,4,3, 3,2,0,3, 1,1,0,2},
        { 26,21,21, 7,12,17,28,21,20,22,25,26, 6,11,10,16,
          16,16, 4,12, 7,18,28,27,30,20,12,14,16,19,17, 6,
          12,12, 8, 0, 3,13,18,15,18,26,20,26,26,32,28,21,
          0, 0, 0, 0, 2, 6, 6, 4, 2, 8, 6, 8,15,10,10,24,
          20,21, 9, 7, 3, 6,15,16,17,22,17,22,17,18,14, 7,
          18, 6,16,12,12,11,17,15,18,18,10,12,27,26,22,18,
          27,28,12,10, 7, 3, 8,13, 8,12,14,16,26,24,24,24,
          9, 9, 6, 0, 0, 0, 2, 6, 0, 0, 0, 0, 4, 8, 8,16,
          26,24,17, 7, 2, 8,11,10,30,24,30,28,32,33,30,24,
          20,11,16,12, 7, 9,17,13,20,14,16,18,31,36,33,29,
          28,25,19, 9, 6,13,20,19, 2, 8, 6, 8,17,17,15,25,
          12,15, 5, 3, 2, 6, 7, 7, 0, 0, 0, 0, 6, 2, 2, 6,
          14,16, 7, 5, 1, 3, 3, 2,20,28,12,20,13,20,20,19,
          9, 4,10, 4, 0, 4, 8, 6, 4,16,12,16,12,18,18,15,
          11,12, 6, 4, 2, 8,10, 7, 0, 0, 0, 0, 9,14,14,14,
          3, 4, 1, 1, 1, 3, 3, 2, 0, 0, 0, 0, 2, 4, 4, 8});
}

struct DepthwiseConvolution2dWeightsPerChannelQuant4_3_2Fixture : DepthwiseConvolution2dFixture2
{
    DepthwiseConvolution2dWeightsPerChannelQuant4_3_2Fixture()
    : DepthwiseConvolution2dFixture2("[ 1, 2, 2, 2 ]",            // inputShape
                                     "[ 1, 2, 2, 4 ]",           // outputShape
                                     "[ 1, 3, 3, 4 ]",           // filterShape
                                     // filter data is [ 0,1,2,3,4,5,6,7,8,
                                     //                  0,1,2,3,4,5,6,7,8,
                                     //                  0,1,2,3,4,5,6,7,8,
                                     //                  0,1,2,3,4,5,6,7,8 ]
                                     //                  quantized per channel with q_dim=3
                                     "[0, 5,20, 9,16,25,60,21,32,"
                                     " 0,10, 6,12,20,50,18,28,40,"
                                     " 0, 3, 8,15,40,15,24,35,80,"
                                     " 0, 4,10,30,12,20,30,70,24]",
                                     "1",                        // stride w and h
                                     "SAME",                     // padding type
                                     "",                         // bias shape
                                     "",                         // bias data
                                     "[ 0.0 ]",                  // filter quantization min values
                                     "[ 255.0 ]",                // filter quantization max values
                                     "[0.25, 0.2, 0.1, 0.3333333333]",   // filter quantization scales
                                     "[ 0, 0, 0, 0]",            // filter quantization zero-points
                                     "3"                         // filter quantized axis
                                                                 // (in case of per channel quantization)
                                    )
    {}
};

// An easy test with M > 1 for debugging
TEST_CASE_FIXTURE(DepthwiseConvolution2dWeightsPerChannelQuant4_3_2Fixture,
                  "ParseDepthwiseConv2DFilterWeightsPerChannelQuant4_3_2")
{
    RunTest<4, armnn::DataType::QAsymmS8>(
        0,
        { 0,1,2,3,4,5,6,7},
        { 38,50,76,92,44,56,66,37,56,50,37,53,62,74,45,61});
}

} // end of TEST_SUITE("TensorflowLiteParser_DepthwiseConvolution2D")
