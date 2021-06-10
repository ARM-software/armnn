//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Mean")
{
struct MeanFixture : public ParserFlatbuffersSerializeFixture
{
    explicit MeanFixture(const std::string &inputShape,
                         const std::string &outputShape,
                         const std::string &axis,
                         const std::string &dataType)
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
                                            dataType: )" + dataType + R"(
                                        }
                                    }]
                                }
                            }
                        }
                    },
                    {
                        layer_type: "MeanLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "MeanLayer",
                                layerType: "Mean",
                                inputSlots: [{
                                    index: 0,
                                    connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                }],
                                outputSlots: [{
                                    index: 0,
                                    tensorInfo: {
                                        dimensions: )" + outputShape + R"(,
                                        dataType: )" + dataType + R"(
                                    }
                                }]
                            },
                            descriptor: {
                                axis: )" + axis + R"(,
                                keepDims: true
                            }
                        }
                    },
                    {
                        layer_type: "OutputLayer",
                        layer: {
                            base:{
                                layerBindingId: 2,
                                base: {
                                    index: 2,
                                    layerName: "OutputLayer",
                                    layerType: "Output",
                                    inputSlots: [{
                                        index: 0,
                                        connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [{
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + outputShape + R"(,
                                            dataType: )" + dataType + R"(
                                        },
                                    }],
                                }
                            }
                        },
                    }
                ]
            }
        )";
        Setup();
    }
};

struct SimpleMeanFixture : MeanFixture
{
    SimpleMeanFixture()
        : MeanFixture("[ 1, 1, 3, 2 ]",     // inputShape
                      "[ 1, 1, 1, 2 ]",     // outputShape
                      "[ 2 ]",              // axis
                      "Float32")            // dataType
    {}
};

TEST_CASE_FIXTURE(SimpleMeanFixture, "SimpleMean")
{
    RunTest<4, armnn::DataType::Float32>(
         0,
         {{"InputLayer",  { 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f }}},
         {{"OutputLayer", { 2.0f, 2.0f }}});
}

}