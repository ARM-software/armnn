//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Normalization")
{
struct NormalizationFixture : public ParserFlatbuffersSerializeFixture
{
    explicit NormalizationFixture(const std::string &inputShape,
        const std::string & outputShape,
        const std::string &dataType,
        const std::string &normAlgorithmChannel,
        const std::string &normAlgorithmMethod,
        const std::string &dataLayout)
    {
        m_JsonString = R"(
        {
            inputIds: [0],
            outputIds: [2],
            layers: [{
                layer_type: "InputLayer",
                layer: {
                    base: {
                        layerBindingId: 0,
                        base: {
                            index: 0,
                            layerName: "InputLayer",
                            layerType: "Input",
                            inputSlots: [{
                                index: 0,
                                connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                }],
                            outputSlots: [{
                                index: 0,
                                tensorInfo: {
                                    dimensions: )" + inputShape + R"(,
                                    dataType: )" + dataType + R"(,
                                    quantizationScale: 0.5,
                                    quantizationOffset: 0
                                    },
                                }]
                            },
                        }
                    },
                },
            {
            layer_type: "NormalizationLayer",
            layer : {
                base: {
                    index:1,
                    layerName: "NormalizationLayer",
                    layerType: "Normalization",
                    inputSlots: [{
                            index: 0,
                            connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                        }],
                    outputSlots: [{
                        index: 0,
                        tensorInfo: {
                            dimensions: )" + outputShape + R"(,
                            dataType: )" + dataType + R"(
                        },
                        }],
                    },
                descriptor: {
                    normChannelType: )" + normAlgorithmChannel + R"(,
                    normMethodType: )" + normAlgorithmMethod + R"(,
                    normSize: 3,
                    alpha: 1,
                    beta: 1,
                    k: 1,
                    dataLayout: )" + dataLayout + R"(
                    }
                },
            },
            {
            layer_type: "OutputLayer",
            layer: {
                base:{
                    layerBindingId: 0,
                    base: {
                        index: 2,
                        layerName: "OutputLayer",
                        layerType: "Output",
                        inputSlots: [{
                            index: 0,
                            connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                        }],
                        outputSlots: [ {
                            index: 0,
                            tensorInfo: {
                                dimensions: )" + outputShape + R"(,
                                dataType: )" + dataType + R"(
                            },
                        }],
                    }
                }},
            }]
        }
 )";
        SetupSingleInputSingleOutput("InputLayer", "OutputLayer");
    }
};

struct FloatNhwcLocalBrightnessAcrossNormalizationFixture : NormalizationFixture
{
    FloatNhwcLocalBrightnessAcrossNormalizationFixture() : NormalizationFixture("[ 2, 2, 2, 1 ]", "[ 2, 2, 2, 1 ]",
        "Float32", "0", "0", "NHWC") {}
};


TEST_CASE_FIXTURE(FloatNhwcLocalBrightnessAcrossNormalizationFixture, "Float32NormalizationNhwcDataLayout")
{
    RunTest<4, armnn::DataType::Float32>(0, { 1.0f, 2.0f, 3.0f, 4.0f,
                                              5.0f, 6.0f, 7.0f, 8.0f },
                                            { 0.5f, 0.400000006f, 0.300000012f, 0.235294119f,
                                              0.192307696f, 0.16216217f, 0.140000001f, 0.123076923f });
}

struct FloatNchwLocalBrightnessWithinNormalizationFixture : NormalizationFixture
{
    FloatNchwLocalBrightnessWithinNormalizationFixture() : NormalizationFixture("[ 2, 1, 2, 2 ]", "[ 2, 1, 2, 2 ]",
        "Float32", "1", "0", "NCHW") {}
};

TEST_CASE_FIXTURE(FloatNchwLocalBrightnessWithinNormalizationFixture, "Float32NormalizationNchwDataLayout")
{
    RunTest<4, armnn::DataType::Float32>(0, { 1.0f, 2.0f, 3.0f, 4.0f,
                                              5.0f, 6.0f, 7.0f, 8.0f },
                                            { 0.0322581f, 0.0645161f, 0.0967742f, 0.1290323f,
                                              0.0285714f, 0.0342857f, 0.04f, 0.0457143f });
}

}