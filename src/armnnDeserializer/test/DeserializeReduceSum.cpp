//
// Copyright © 2020 Samsung Electronics Co Ltd and Contributors. All rights reserved.
// Copyright © 2021 Arm Ltd and Contributors. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include "../Deserializer.hpp"

#include <string>

TEST_SUITE("Deserializer_ReduceSum")
{
struct ReduceSumFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ReduceSumFixture(const std::string& inputShape,
                              const std::string& outputShape,
                              const std::string& axis,
                              const std::string& dataType)
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
                        layer_type: "ReduceLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "ReduceSumLayer",
                                layerType: "Reduce",
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
                                keepDims: true,
                                reduceOperation: Sum
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

struct SimpleReduceSumFixture : ReduceSumFixture
{
    SimpleReduceSumFixture()
        : ReduceSumFixture("[ 1, 1, 3, 2 ]",     // inputShape
                           "[ 1, 1, 1, 2 ]",     // outputShape
                           "[ 2 ]",              // axis
                           "Float32")            // dataType
    {}
};

TEST_CASE_FIXTURE(SimpleReduceSumFixture, "SimpleReduceSum")
{
    RunTest<4, armnn::DataType::Float32>(
         0,
         {{"InputLayer",  { 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f }}},
         {{"OutputLayer", { 6.0f, 6.0f }}});
}

}
