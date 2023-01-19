//
// Copyright © 2017-2023 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersFixture.hpp"


TEST_SUITE("TensorflowLiteParser_FullyConnected")
{
struct FullyConnectedFixture : public ParserFlatbuffersFixture
{
    explicit FullyConnectedFixture(const std::string& inputShape,
                                   const std::string& outputShape,
                                   const std::string& filterShape,
                                   const std::string& filterData,
                                   const std::string biasShape = "",
                                   const std::string biasData = "",
                                   const std::string dataType = "UINT8",
                                   const std::string weightsDataType = "UINT8",
                                   const std::string biasDataType = "INT32")
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
                            "type": )" + biasDataType + R"(,
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
                "operator_codes": [ { "builtin_code": "FULLY_CONNECTED" } ],
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
                                "scale": [ 1.0 ],
                                "zero_point": [ 0 ],
                            }
                        },
                        {
                            "shape": )" + outputShape + R"(,
                            "type": )" + dataType + R"(,
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
                            "type": )" + weightsDataType + R"(,
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
                            "builtin_options_type": "FullyConnectedOptions",
                            "builtin_options": {
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

struct FullyConnectedWithNoBiasFixture : FullyConnectedFixture
{
    FullyConnectedWithNoBiasFixture()
        : FullyConnectedFixture("[ 1, 4, 1, 1 ]",     // inputShape
                                "[ 1, 1 ]",           // outputShape
                                "[ 1, 4 ]",           // filterShape
                                "[ 2, 3, 4, 5 ]")     // filterData
    {}
};

TEST_CASE_FIXTURE(FullyConnectedWithNoBiasFixture, "FullyConnectedWithNoBias")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
        0,
        { 10, 20, 30, 40 },
        { 400/2 });
}

struct FullyConnectedWithBiasFixture : FullyConnectedFixture
{
    FullyConnectedWithBiasFixture()
        : FullyConnectedFixture("[ 1, 4, 1, 1 ]",     // inputShape
                                "[ 1, 1 ]",           // outputShape
                                "[ 1, 4 ]",           // filterShape
                                "[ 2, 3, 4, 5 ]",     // filterData
                                "[ 1 ]",              // biasShape
                                "[ 10, 0, 0, 0 ]" )   // biasData
    {}
};

TEST_CASE_FIXTURE(FullyConnectedWithBiasFixture, "ParseFullyConnectedWithBias")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
        0,
        { 10, 20, 30, 40 },
        { (400+10)/2 });
}

struct FullyConnectedWithBiasMultipleOutputsFixture : FullyConnectedFixture
{
    FullyConnectedWithBiasMultipleOutputsFixture()
            : FullyConnectedFixture("[ 1, 4, 2, 1 ]",     // inputShape
                                    "[ 2, 1 ]",           // outputShape
                                    "[ 1, 4 ]",           // filterShape
                                    "[ 2, 3, 4, 5 ]",     // filterData
                                    "[ 1 ]",              // biasShape
                                    "[ 10, 0, 0, 0 ]" )   // biasData
    {}
};

TEST_CASE_FIXTURE(FullyConnectedWithBiasMultipleOutputsFixture, "FullyConnectedWithBiasMultipleOutputs")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
            0,
            { 1, 2, 3, 4, 10, 20, 30, 40 },
            { (40+10)/2, (400+10)/2 });
}

struct DynamicFullyConnectedWithBiasMultipleOutputsFixture : FullyConnectedFixture
{
    DynamicFullyConnectedWithBiasMultipleOutputsFixture()
        : FullyConnectedFixture("[ 1, 4, 2, 1 ]",     // inputShape
                                "[ ]",               // outputShape
                                "[ 1, 4 ]",           // filterShape
                                "[ 2, 3, 4, 5 ]",     // filterData
                                "[ 1 ]",              // biasShape
                                "[ 10, 0, 0, 0 ]" )   // biasData
    { }
};

TEST_CASE_FIXTURE(
    DynamicFullyConnectedWithBiasMultipleOutputsFixture, "DynamicFullyConnectedWithBiasMultipleOutputs")
{
    RunTest<2,
            armnn::DataType::QAsymmU8,
            armnn::DataType::QAsymmU8>(0,
                                      { { "inputTensor", { 1, 2, 3, 4, 10, 20, 30, 40} } },
                                      { { "outputTensor", { (40+10)/2, (400+10)/2 } } },
                                      true);
}


struct FullyConnectedNonConstWeightsFixture : public ParserFlatbuffersFixture
{
    explicit FullyConnectedNonConstWeightsFixture(const std::string& inputShape,
                                                  const std::string& outputShape,
                                                  const std::string& filterShape,
                                                  const std::string biasShape = "")
    {
        std::string inputTensors = "[ 0, 1 ]";
        std::string biasTensor = "";
        std::string biasBuffer = "";
        std::string outputs = "2";
        if (biasShape.size() > 0)
        {
            inputTensors = "[ 0, 1, 2 ]";
            biasTensor = R"(
                       {
                      "shape": )" + biasShape + R"(,
                      "type": "INT32",
                      "buffer": 2,
                      "name": "bias",
                      "quantization": {
                        "scale": [ 1.0 ],
                        "zero_point": [ 0 ],
                        "details_type": 0,
                        "quantized_dimension": 0
                      },
                      "is_variable": true
                    }, )";

            biasBuffer = R"(,{ "data": [] } )";
            outputs = "3";
        }
        m_JsonString = R"(
            {
              "version": 3,
              "operator_codes": [
                {
                  "builtin_code": "FULLY_CONNECTED",
                  "version": 1
                }
              ],
              "subgraphs": [
                {
                  "tensors": [
                    {
                      "shape": )" + inputShape + R"(,
                      "type": "INT8",
                      "buffer": 0,
                      "name": "input_0",
                      "quantization": {
                        "scale": [ 1.0 ],
                        "zero_point": [ 0 ],
                        "details_type": 0,
                        "quantized_dimension": 0
                      },
                    },
                    {
                      "shape": )" + filterShape + R"(,
                      "type": "INT8",
                      "buffer": 1,
                      "name": "weights",
                      "quantization": {
                        "scale": [ 1.0 ],
                        "zero_point": [ 0 ],
                        "details_type": 0,
                        "quantized_dimension": 0
                      },
                    },
                    )" + biasTensor + R"(
                    {
                      "shape": )" + outputShape + R"(,
                      "type": "INT8",
                      "buffer": 0,
                      "name": "output",
                      "quantization": {
                        "scale": [
                          2.0
                        ],
                        "zero_point": [
                          0
                        ],
                        "details_type": 0,
                        "quantized_dimension": 0
                      },
                    }
                  ],
                  "inputs": )" + inputTensors + R"(,
                  "outputs": [ )" + outputs + R"( ],
                  "operators": [
                    {
                      "opcode_index": 0,
                      "inputs": )" + inputTensors + R"(,
                      "outputs": [ )" + outputs + R"( ],
                      "builtin_options_type": "FullyConnectedOptions",
                      "builtin_options": {
                        "fused_activation_function": "NONE",
                        "weights_format": "DEFAULT",
                        "keep_num_dims": false,
                        "asymmetric_quantize_inputs": false
                      },
                      "custom_options_format": "FLEXBUFFERS"
                    }
                  ]
                }
              ],
              "description": "ArmnnDelegate: FullyConnected Operator Model",
              "buffers": [
                {
                  "data": []
                },
                {
                  "data": []
                }
                )" + biasBuffer + R"(
              ]
            }
            )";
        Setup();
    }
};

struct FullyConnectedNonConstWeights : FullyConnectedNonConstWeightsFixture
{
    FullyConnectedNonConstWeights()
            : FullyConnectedNonConstWeightsFixture("[ 1, 4, 1, 1 ]",     // inputShape
                                                   "[ 1, 1 ]",           // outputShape
                                                   "[ 1, 4 ]",           // filterShape
                                                   "[ 1 ]" )             // biasShape

    {}
};

TEST_CASE_FIXTURE(FullyConnectedNonConstWeights, "ParseFullyConnectedNonConstWeights")
{
    RunTest<2, armnn::DataType::QAsymmS8,
            armnn::DataType::Signed32,
            armnn::DataType::QAsymmS8>(
            0,
            {{{"input_0", { 1, 2, 3, 4 }},{"weights", { 2, 3, 4, 5 }}}},
            {{"bias", { 10 }}},
            {{"output", { 25 }}});
}

struct FullyConnectedNonConstWeightsNoBias : FullyConnectedNonConstWeightsFixture
{
    FullyConnectedNonConstWeightsNoBias()
            : FullyConnectedNonConstWeightsFixture("[ 1, 4, 1, 1 ]",     // inputShape
                                                   "[ 1, 1 ]",           // outputShape
                                                   "[ 1, 4 ]")           // filterShape

    {}
};

TEST_CASE_FIXTURE(FullyConnectedNonConstWeightsNoBias, "ParseFullyConnectedNonConstWeightsNoBias")
{
    RunTest<2, armnn::DataType::QAsymmS8,
            armnn::DataType::QAsymmS8>(
            0,
            {{{"input_0", { 1, 2, 3, 4 }},{"weights", { 2, 3, 4, 5 }}}},
            {{"output", { 20 }}});
}

struct FullyConnectedWeightsBiasFloat : FullyConnectedFixture
{
    FullyConnectedWeightsBiasFloat()
            : FullyConnectedFixture("[ 1, 4, 1, 1 ]",     // inputShape
                                    "[ 1, 1, 1, 1 ]",     // outputShape
                                    "[ 1, 4 ]",           // filterShape
                                    "[ 2, 3, 4, 5 ]",     // filterData
                                    "[ 1 ]",              // biasShape
                                    "[ 10, 0, 0, 0 ]",    // filterShape
                                    "FLOAT32",            // input and output dataType
                                    "INT8",               // weights dataType
                                    "FLOAT32")            // bias dataType
    {}
};

TEST_CASE_FIXTURE(FullyConnectedWeightsBiasFloat, "FullyConnectedWeightsBiasFloat")
{
    RunTest<4, armnn::DataType::Float32>(
            0,
            { 10, 20, 30, 40 },
            { 400 });
}

}
