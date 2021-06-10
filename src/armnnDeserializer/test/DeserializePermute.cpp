//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Permute")
{
struct PermuteFixture : public ParserFlatbuffersSerializeFixture
{
    explicit PermuteFixture(const std::string &inputShape,
                            const std::string &dimMappings,
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
                        layer_type: "PermuteLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "PermuteLayer",
                                layerType: "Permute",
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
                                dimMappings: )" + dimMappings + R"(,
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

struct SimplePermute2DFixture : PermuteFixture
{
    SimplePermute2DFixture() : PermuteFixture("[ 2, 3 ]",
                                              "[ 1, 0 ]",
                                              "[ 3, 2 ]",
                                              "QuantisedAsymm8") {}
};

TEST_CASE_FIXTURE(SimplePermute2DFixture, "SimplePermute2DQuantisedAsymm8")
{
    RunTest<2, armnn::DataType::QAsymmU8>(0,
                                                 { 1, 2, 3, 4, 5, 6 },
                                                 { 1, 4, 2, 5, 3, 6 });
}

struct SimplePermute4DFixture : PermuteFixture
{
    SimplePermute4DFixture() : PermuteFixture("[ 1, 2, 3, 4 ]",
                                              "[ 3, 2, 1, 0 ]",
                                              "[ 4, 3, 2, 1 ]",
                                              "QuantisedAsymm8") {}
};

TEST_CASE_FIXTURE(SimplePermute4DFixture, "SimplePermute4DQuantisedAsymm8")
{
    RunTest<4, armnn::DataType::QAsymmU8>(0,
                                                 {  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12,
                                                   13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24 },
                                                 {  1, 13,  5, 17,  9, 21,  2, 14,  6, 18, 10, 22,
                                                    3, 15,  7, 19, 11, 23,  4, 16,  8, 20, 12, 24 });
}

}
