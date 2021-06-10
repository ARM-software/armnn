//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_L2Normalization")
{
struct L2NormalizationFixture : public ParserFlatbuffersSerializeFixture
{
    explicit L2NormalizationFixture(const std::string &inputShape,
                                    const std::string &outputShape,
                                    const std::string &dataType,
                                    const std::string &dataLayout,
                                    const std::string epsilon)
    {
        m_JsonString = R"(
    {
        inputIds: [0],
        outputIds: [2],
        layers: [
           {
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
                                dataType: ")" + dataType + R"(",
                                quantizationScale: 0.5,
                                quantizationOffset: 0
                                },
                            }]
                        },
                    }
                },
            },
        {
        layer_type: "L2NormalizationLayer",
        layer : {
            base: {
                index:1,
                layerName: "L2NormalizationLayer",
                layerType: "L2Normalization",
                inputSlots: [{
                        index: 0,
                        connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                   }],
                outputSlots: [{
                    index: 0,
                    tensorInfo: {
                        dimensions: )" + outputShape + R"(,
                        dataType: ")" + dataType + R"("
                    },
                    }],
                },
            descriptor: {
                dataLayout: ")" + dataLayout + R"(",
                eps: )" + epsilon + R"(
                },
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
                            dataType: ")" + dataType + R"("
                        },
                    }],
                }
            }},
        }]
    }
)";
        Setup();
    }
};

struct L2NormFixture : L2NormalizationFixture
{
    // Using a non standard epsilon value of 1e-8
    L2NormFixture():L2NormalizationFixture("[ 1, 3, 1, 1 ]",
                                           "[ 1, 3, 1, 1 ]",
                                           "Float32",
                                           "NCHW",
                                           "0.00000001"){}
};

TEST_CASE_FIXTURE(L2NormFixture, "L2NormalizationFloat32")
{
    // 1 / sqrt(1^2 + 2^2 + 3^2)
    const float approxInvL2Norm = 0.267261f;

    RunTest<4, armnn::DataType::Float32>(0,
                                         {{"InputLayer", { 1.0f, 2.0f, 3.0f }}},
                                         {{"OutputLayer",{ 1.0f * approxInvL2Norm,
                                                           2.0f * approxInvL2Norm,
                                                           3.0f * approxInvL2Norm }}});
}

TEST_CASE_FIXTURE(L2NormFixture, "L2NormalizationEpsilonLimitFloat32")
{
    // 1 / sqrt(1e-8)
    const float approxInvL2Norm = 10000;

    RunTest<4, armnn::DataType::Float32>(0,
                                         {{"InputLayer", { 0.00000001f, 0.00000002f, 0.00000003f }}},
                                         {{"OutputLayer",{ 0.00000001f * approxInvL2Norm,
                                                           0.00000002f * approxInvL2Norm,
                                                           0.00000003f * approxInvL2Norm }}});
}

}
