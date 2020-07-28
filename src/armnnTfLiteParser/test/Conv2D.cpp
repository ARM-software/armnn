//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersFixture.hpp"
#include "../TfLiteParser.hpp"
#include <sstream>

BOOST_AUTO_TEST_SUITE(TensorflowLiteParser)

struct SimpleConv2DFixture : public ParserFlatbuffersFixture
{
    explicit SimpleConv2DFixture()
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "CONV_2D" } ],
                "subgraphs": [ {
                    "tensors": [
                        {
                            "shape": [ 1, 3, 3, 1 ],
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
                            "shape": [ 1, 1, 1, 1 ],
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
                            "shape": [ 1, 3, 3, 1 ],
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
                            "builtin_options_type": "Conv2DOptions",
                            "builtin_options": {
                                "padding": "VALID",
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
                    { "data": [ 2,1,0,  6,2,1, 4,1,2 ], },
                    { },
                ]
            }
        )";
        SetupSingleInputSingleOutput("inputTensor", "outputTensor");
    }
};

BOOST_FIXTURE_TEST_CASE( ParseSimpleConv2D, SimpleConv2DFixture )
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        {
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        },
        // because of the output scaling we need to take half of the values
        {
            (1*2 + 2*1 + 3*0 +
             4*6 + 5*2 + 6*1 +
             7*4 + 8*1 + 9*2) /2
        });
}

struct Conv2DWithBiasesFixture : public ParserFlatbuffersFixture
{
    explicit Conv2DWithBiasesFixture(const std::string & inputShape,
                                     const std::string & outputShape,
                                     const std::string & filterShape,
                                     const std::string & filterData,
                                     const std::string & biasShape,
                                     const std::string & biasData,
                                     const std::string & strides,
                                     const std::string & activation="NONE",
                                     const std::string & filterScale="1.0",
                                     const std::string & filterZeroPoint="0",
                                     const std::string & outputScale="2.0",
                                     const std::string & outputZeroPoint="0")
    {
        m_JsonString = R"(
            {
                "version": 3,
                "operator_codes": [ { "builtin_code": "CONV_2D" } ],
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
                            "builtin_options_type": "Conv2DOptions",
                            "builtin_options": {
                                "padding": "SAME",
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

struct SimpleConv2DWithBiasesFixture : Conv2DWithBiasesFixture
{
    SimpleConv2DWithBiasesFixture()
    : Conv2DWithBiasesFixture("[ 1, 2, 2, 1 ]",    // inputShape
                              "[ 1, 2, 2, 1 ]",    // outputShape
                              "[ 1, 2, 2, 1 ]",    // filterShape
                              "[ 2,1, 0,6 ]",      // filterData
                              "[ 1 ]",             // biasShape
                              "[ 10, 0, 0, 0 ]",   // biasData
                              "1")                 // stride w and h
    {}
};

BOOST_FIXTURE_TEST_CASE( ParseConv2DWithBias, SimpleConv2DWithBiasesFixture )
{
    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        {
            1, 2,
            3, 4,
        },
        // because of the output scaling we need to take half of the values
        {
            (1*2 + 2*1 + 3*0 + 4*6 + 10)/2,
            (2*2 + 0*1 + 4*0 + 0*6 + 10)/2,
            (3*2 + 4*1 + 0*0 + 0*6 + 10)/2,
            (4*2 + 0*1 + 0*0 + 0*6 + 10)/2
        });
}

struct DynamicConv2DWithBiasesFixture : Conv2DWithBiasesFixture
{
    DynamicConv2DWithBiasesFixture()
        : Conv2DWithBiasesFixture("[ 1, 2, 2, 1 ]",    // inputShape
                                  "[ ]",              // outputShape
                                  "[ 1, 2, 2, 1 ]",    // filterShape
                                  "[ 2,1, 0,6 ]",      // filterData
                                  "[ 1 ]",             // biasShape
                                  "[ 10, 0, 0, 0 ]",   // biasData
                                  "1")                 // stride w and h
    {}
};

BOOST_FIXTURE_TEST_CASE( ParseDynamicConv2DWithBias, DynamicConv2DWithBiasesFixture )
{
    RunTest<4,
        armnn::DataType::QAsymmU8,
        armnn::DataType::QAsymmU8>(0,
                                   { { "inputTensor", { 1, 2, 3, 4, } } },
                                   { { "outputTensor", {   (1*2 + 2*1 + 3*0 + 4*6 + 10)/2,
                                                           (2*2 + 0*1 + 4*0 + 0*6 + 10)/2,
                                                           (3*2 + 4*1 + 0*0 + 0*6 + 10)/2,
                                                           (4*2 + 0*1 + 0*0 + 0*6 + 10)/2} } },
                                   true);
}

struct Conv2DShapeTestFixture : Conv2DWithBiasesFixture
{
    static std::string GenerateInts(unsigned int n)
    {
        std::stringstream ss;
        ss << " [ ";
        for( unsigned int i=0; i<n; ++i ) {
            if (i > 0 )
            {
                ss << " , ";
            }
            ss << " " << (i%256);
        }
        ss << " ] ";
        return ss.str();
    }

    Conv2DShapeTestFixture()
    : Conv2DWithBiasesFixture("[ 1, 224, 224, 3 ]",    // inputShape
                              "[ 1, 112, 112, 32 ]",   // outputShape
                              "[ 32, 3, 3, 3 ]",       // filterShape
                              GenerateInts(32*3*3*3),  // filterData
                              "[ 32 ]",                // biasShape
                              GenerateInts(32*4),      // biasData
                              "2")                     // stride w and h
    {}
};

BOOST_FIXTURE_TEST_CASE( ParseConv2D_112x112_out, Conv2DShapeTestFixture )
{
}

struct ReluConv2DWithBiasesFixture : Conv2DWithBiasesFixture
{
    ReluConv2DWithBiasesFixture()
    : Conv2DWithBiasesFixture("[ 1, 2, 2, 1 ]",    // inputShape
                              "[ 1, 2, 2, 1 ]",    // outputShape
                              "[ 1, 2, 2, 1 ]",    // filterShape
                              "[ 2,1, 0,6 ]",      // filterData
                              "[ 1 ]",             // biasShape
                              "[ 16, 0, 0, 0 ]",   // biasData
                              "1",                 // stride w and h
                              "RELU",              // activation
                              "1.0",               // filter scale
                              "4",                 // filter zero point
                              "2.0",               // output scale
                              "20")                // output zero point
    {}
};

BOOST_FIXTURE_TEST_CASE( ParseConv2DAndReluWithBias, ReluConv2DWithBiasesFixture )
{
    uint8_t bias = 16;
    uint8_t outZero = 20;
    uint8_t fz = 4; // filter zero point

    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        {
            1, 2,
            4, 8,
        },
        // factors to consider:
        // - the filter zero point is non zero, hence the (x-fz)
        // - the output scale is 2 hence the /2
        // - output zero point is non zero, hence the +outZero
        // - RELU cuts negative values and then we add the output zero point
        {
            std::max(outZero, static_cast<uint8_t>((1*(2-fz) + 2*(1-fz) + 4*(0-fz) + 8*(6-fz) + bias)/2 + outZero)),
            std::max(outZero, static_cast<uint8_t>((2*(2-fz) + 0*(1-fz) + 8*(0-fz) + 0*(6-fz) + bias)/2 + outZero)),
            std::max(outZero, static_cast<uint8_t>((4*(2-fz) + 8*(1-fz) + 0*(0-fz) + 0*(6-fz) + bias)/2 + outZero)),
            std::max(outZero, static_cast<uint8_t>((8*(2-fz) + 0*(1-fz) + 0*(0-fz) + 0*(6-fz) + bias)/2 + outZero))
        });
}

struct Relu6Conv2DWithBiasesFixture : Conv2DWithBiasesFixture
{
    Relu6Conv2DWithBiasesFixture()
    : Conv2DWithBiasesFixture("[ 1, 2, 2, 1 ]",    // inputShape
                              "[ 1, 2, 2, 1 ]",    // outputShape
                              "[ 1, 2, 2, 1 ]",    // filterShape
                              "[ 2,1, 0,6 ]",      // filterData
                              "[ 1 ]",             // biasShape
                              "[ 0, 0, 0, 0 ]",    // biasData
                              "1",                 // stride w and h
                              "RELU6",             // activation
                              "1.0",               // filter scale
                              "0",                 // filter zero point
                              "2.0",               // output scale
                              "0")                 // output zero point
    {}
};

BOOST_FIXTURE_TEST_CASE( ParseConv2DAndRelu6WithBias, Relu6Conv2DWithBiasesFixture )
{
    uint8_t relu6Min = 6 / 2; // divide by output scale

    RunTest<4, armnn::DataType::QAsymmU8>(
        0,
        {
            1, 2,
            4, 1,
        },
        // factors to consider:
        // - the output scale is 2 hence the /2
        // - RELU6 cuts output values at +6
        {
            std::min(relu6Min, static_cast<uint8_t>((1*2 + 2*1 + 4*0 + 1*6)/2)),
            std::min(relu6Min, static_cast<uint8_t>((2*2 + 0*1 + 1*0 + 0*6)/2)),
            std::min(relu6Min, static_cast<uint8_t>((4*2 + 1*1 + 0*0 + 0*6)/2)),
            std::min(relu6Min, static_cast<uint8_t>((1*2 + 0*1 + 0*0 + 0*6)/2))
        });
}

BOOST_AUTO_TEST_SUITE_END()
