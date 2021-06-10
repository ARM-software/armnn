//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_BatchToSpaceND")
{
struct BatchToSpaceNdFixture : public ParserFlatbuffersSerializeFixture
{
    explicit BatchToSpaceNdFixture(const std::string &inputShape,
                                   const std::string &blockShape,
                                   const std::string &crops,
                                   const std::string &dataLayout,
                                   const std::string &outputShape,
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
                        layer_type: "BatchToSpaceNdLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "BatchToSpaceNdLayer",
                                layerType: "BatchToSpaceNd",
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
                                blockShape: )" + blockShape + R"(,
                                crops: )" + crops + R"(,
                                dataLayout: )" + dataLayout + R"(,
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
        SetupSingleInputSingleOutput("InputLayer", "OutputLayer");
    }
};

struct SimpleBatchToSpaceNdFixture : BatchToSpaceNdFixture
{
    SimpleBatchToSpaceNdFixture() : BatchToSpaceNdFixture("[ 4, 2, 2, 1 ]",
                                                          "[ 2, 2 ]",
                                                          "[ 0, 0, 0, 0 ]",
                                                          "NHWC",
                                                          "[ 1, 4, 4, 1 ]",
                                                          "Float32") {}
};

TEST_CASE_FIXTURE(SimpleBatchToSpaceNdFixture, "SimpleBatchToSpaceNdFloat32")
{
    RunTest<4, armnn::DataType::Float32>(0,
                                         {
                                             1.0f, 3.0f,  9.0f, 11.0f,
                                             2.0f, 4.0f, 10.0f, 12.0f,
                                             5.0f, 7.0f, 13.0f, 15.0f,
                                             6.0f, 8.0f, 14.0f, 16.0f
                                         },
                                         {
                                              1.0f,  2.0f,  3.0f,  4.0f,
                                              5.0f,  6.0f,  7.0f,  8.0f,
                                              9.0f, 10.0f, 11.0f, 12.0f,
                                             13.0f, 14.0f, 15.0f, 16.0f
                                         });
}

}
