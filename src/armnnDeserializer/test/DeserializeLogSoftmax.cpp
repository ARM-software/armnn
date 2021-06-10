//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

TEST_SUITE("Deserializer_LogSoftmax")
{
struct LogSoftmaxFixture : public ParserFlatbuffersSerializeFixture
{
    explicit LogSoftmaxFixture(const std::string &shape,
                               const std::string &beta,
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
                                connection: { sourceLayerIndex:0, outputSlotIndex:0 },
                                }],
                            outputSlots: [{
                                index: 0,
                                tensorInfo: {
                                    dimensions: )" + shape + R"(,
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
            layer_type: "LogSoftmaxLayer",
            layer : {
                base: {
                    index:1,
                    layerName: "LogSoftmaxLayer",
                    layerType: "LogSoftmax",
                    inputSlots: [{
                            index: 0,
                            connection: { sourceLayerIndex:0, outputSlotIndex:0 },
                        }],
                    outputSlots: [{
                        index: 0,
                        tensorInfo: {
                            dimensions: )" + shape + R"(,
                            dataType: ")" + dataType + R"("
                        },
                        }],
                    },
                descriptor: {
                    beta: ")" + beta + R"(",
                    axis: )" + axis + R"(
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
                            connection: { sourceLayerIndex:1, outputSlotIndex:0 },
                        }],
                        outputSlots: [ {
                            index: 0,
                            tensorInfo: {
                                dimensions: )" + shape + R"(,
                                dataType: ")" + dataType + R"("
                            },
                        }],
                    }
                }},
            }]
        })";
        SetupSingleInputSingleOutput("InputLayer", "OutputLayer");
    }
};

struct LogSoftmaxFloat32Fixture : LogSoftmaxFixture
{
    LogSoftmaxFloat32Fixture() :
        LogSoftmaxFixture("[ 1, 1, 2, 4 ]", // inputShape
                          "1.0",            // beta
                          "3",              // axis
                          "Float32")        // dataType
    {}
};

TEST_CASE_FIXTURE(LogSoftmaxFloat32Fixture, "LogSoftmaxFloat32")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        {
            0.f, -6.f,  2.f, 4.f,
            3.f, -2.f, 10.f, 1.f
        },
        {
            -4.14297f, -10.14297f, -2.14297f, -0.14297f,
            -7.00104f, -12.00104f, -0.00105f, -9.00104f
        });
}

}
