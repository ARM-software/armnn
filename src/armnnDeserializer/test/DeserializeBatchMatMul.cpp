//
// Copyright © 2022 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <doctest/doctest.h>

#include <string>

TEST_SUITE("Deserializer_BatchMatMul")
{
struct BatchMatMulFixture : public ParserFlatbuffersSerializeFixture
{
    explicit BatchMatMulFixture(const std::string& inputXShape,
                                const std::string& inputYShape,
                                const std::string& outputShape,
                                const std::string& dataType)
    {
        m_JsonString = R"(
            {
                inputIds:[
                    0,
                    1
                ],
                outputIds:[
                    3
                ],
                layers:[
                    {
                        layer_type:"InputLayer",
                        layer:{
                            base:{
                                layerBindingId:0,
                                base:{
                                    index:0,
                                    layerName:"InputXLayer",
                                    layerType:"Input",
                                    inputSlots:[
                                        {
                                            index:0,
                                            connection:{
                                                sourceLayerIndex:0,
                                                outputSlotIndex:0
                                            },

                                        }
                                    ],
                                    outputSlots:[
                                        {
                                            index:0,
                                            tensorInfo:{
                                                dimensions:)" + inputXShape + R"(,
                                                dataType:)" + dataType + R"(
                                            },

                                        }
                                    ],

                                },

                            }
                        },

                    },
                    {
                        layer_type:"InputLayer",
                        layer:{
                            base:{
                                layerBindingId:1,
                                base:{
                                    index:1,
                                    layerName:"InputYLayer",
                                    layerType:"Input",
                                    inputSlots:[
                                        {
                                            index:0,
                                            connection:{
                                                sourceLayerIndex:0,
                                                outputSlotIndex:0
                                            },

                                        }
                                    ],
                                    outputSlots:[
                                        {
                                            index:0,
                                            tensorInfo:{
                                                dimensions:)" + inputYShape + R"(,
                                                dataType:)" + dataType + R"(
                                            },

                                        }
                                    ],

                                },

                            }
                        },

                    },
                    {
                        layer_type:"BatchMatMulLayer",
                        layer:{
                            base:{
                                index:2,
                                layerName:"BatchMatMulLayer",
                                layerType:"BatchMatMul",
                                inputSlots:[
                                    {
                                        index:0,
                                        connection:{
                                            sourceLayerIndex:0,
                                            outputSlotIndex:0
                                        },

                                    },
                                    {
                                        index:1,
                                        connection:{
                                            sourceLayerIndex:1,
                                            outputSlotIndex:0
                                        },

                                    }
                                ],
                                outputSlots:[
                                    {
                                        index:0,
                                        tensorInfo:{
                                            dimensions:)" + outputShape + R"(,
                                            dataType:)" + dataType + R"(
                                        },

                                    }
                                ],

                            },
                            descriptor:{
                                transposeX:false,
                                transposeY:false,
                                adjointX:false,
                                adjointY:false,
                                dataLayoutX:NHWC,
                                dataLayoutY:NHWC
                            }
                        },

                    },
                    {
                        layer_type:"OutputLayer",
                        layer:{
                            base:{
                                layerBindingId:0,
                                base:{
                                    index:3,
                                    layerName:"OutputLayer",
                                    layerType:"Output",
                                    inputSlots:[
                                        {
                                            index:0,
                                            connection:{
                                                sourceLayerIndex:2,
                                                outputSlotIndex:0
                                            },

                                        }
                                    ],
                                    outputSlots:[
                                        {
                                            index:0,
                                            tensorInfo:{
                                                dimensions:)" + outputShape + R"(,
                                                dataType:)" + dataType + R"(
                                            },

                                        }
                                    ],

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

struct SimpleBatchMatMulFixture : BatchMatMulFixture
{
    SimpleBatchMatMulFixture()
        : BatchMatMulFixture("[ 1, 2, 2, 1 ]",
                             "[ 1, 2, 2, 1 ]",
                             "[ 1, 2, 2, 1 ]",
                             "Float32")
    {}
};

TEST_CASE_FIXTURE(SimpleBatchMatMulFixture, "SimpleBatchMatMulTest")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        {{"InputXLayer", { 1.0f, 2.0f, 3.0f, 4.0f }},
         {"InputYLayer", { 5.0f, 6.0f, 7.0f, 8.0f }}},
        {{"OutputLayer", { 19.0f, 22.0f, 43.0f, 50.0f }}});
}

}