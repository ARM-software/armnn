//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <doctest/doctest.h>

#include <string>

TEST_SUITE("Deserializer_Abs")
{
    struct AbsFixture : public ParserFlatbuffersSerializeFixture
    {
        explicit AbsFixture(const std::string &inputShape,
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
                        layer_type: "AbsLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "AbsLayer",
                                layerType: "Abs",
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

    struct SimpleAbsFixture : AbsFixture
    {
        SimpleAbsFixture()
                : AbsFixture("[ 1, 2, 2, 2 ]",     // inputShape
                              "[ 1, 2, 2, 2 ]",     // outputShape
                              "Float32")            // dataType
        {}
    };

    TEST_CASE_FIXTURE(SimpleAbsFixture, "SimpleAbsTest")
    {
        RunTest<4, armnn::DataType::Float32>(
                0,
                {{"InputLayer",  { -100.0f, -50.5f, -25.9999f, -0.5f , 0.0f, 1.5555f, 25.5f, 100.0f }}},
                {{"OutputLayer", { 100.0f, 50.5f, 25.9999f, 0.5f , 0.0f, 1.5555f, 25.5f, 100.0f }}});
    }

}