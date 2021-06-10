//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_InstanceNormalization")
{
struct InstanceNormalizationFixture : public ParserFlatbuffersSerializeFixture
{
    explicit InstanceNormalizationFixture(const std::string &inputShape,
                                          const std::string &outputShape,
                                          const std::string &gamma,
                                          const std::string &beta,
                                          const std::string &epsilon,
                                          const std::string &dataType,
                                          const std::string &dataLayout)
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
        layer_type: "InstanceNormalizationLayer",
        layer : {
            base: {
                index:1,
                layerName: "InstanceNormalizationLayer",
                layerType: "InstanceNormalization",
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
                gamma: ")" + gamma + R"(",
                beta: ")" + beta + R"(",
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
        SetupSingleInputSingleOutput("InputLayer", "OutputLayer");
    }
};

struct InstanceNormalizationFloat32Fixture : InstanceNormalizationFixture
{
    InstanceNormalizationFloat32Fixture():InstanceNormalizationFixture("[ 2, 2, 2, 2 ]",
                                                                       "[ 2, 2, 2, 2 ]",
                                                                       "1.0",
                                                                       "0.0",
                                                                       "0.0001",
                                                                       "Float32",
                                                                       "NHWC") {}
};

TEST_CASE_FIXTURE(InstanceNormalizationFloat32Fixture, "InstanceNormalizationFloat32")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
         {
             0.f,  1.f,
             0.f,  2.f,

             0.f,  2.f,
             0.f,  4.f,

             1.f, -1.f,
            -1.f,  2.f,

            -1.f, -2.f,
             1.f,  4.f
        },
        {
             0.0000000f, -1.1470304f,
             0.0000000f, -0.2294061f,

             0.0000000f, -0.2294061f,
             0.0000000f,  1.6058424f,

             0.9999501f, -0.7337929f,
            -0.9999501f,  0.5241377f,

            -0.9999501f, -1.1531031f,
             0.9999501f,  1.3627582f
        });
}

}
