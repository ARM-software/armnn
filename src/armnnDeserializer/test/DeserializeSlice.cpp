//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"

#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Slice")
{
struct SliceFixture : public ParserFlatbuffersSerializeFixture
{
    explicit SliceFixture(const std::string& inputShape,
                          const std::string& outputShape,
                          const std::string& begin,
                          const std::string& size,
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
                        layer_type: "SliceLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "SliceLayer",
                                layerType: "Slice",
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
                                begin: )" + begin + R"(,
                                size: )" + size + R"(,
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

struct SimpleSliceFixture : SliceFixture
{
    SimpleSliceFixture() : SliceFixture("[ 3, 2, 3, 5 ]", // input shape
                                        "[ 2, 1, 2, 3 ]", // output shape
                                        "[ 1, 0, 1, 2 ]", // begin
                                        "[ 2, 1, 2, 3 ]", // size
                                        "Float32") {}     // data type
};

TEST_CASE_FIXTURE(SimpleSliceFixture, "SimpleSliceFloat32")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        {
            0.f,  1.f,  2.f,  3.f,  4.f,
            5.f,  6.f,  7.f,  8.f,  9.f,
            10.f, 11.f, 12.f, 13.f, 14.f,

            15.f, 16.f, 17.f, 18.f, 19.f,
            20.f, 21.f, 22.f, 23.f, 24.f,
            25.f, 26.f, 27.f, 28.f, 29.f,


            30.f, 31.f, 32.f, 33.f, 34.f,
            35.f, 36.f, 37.f, 38.f, 39.f,
            40.f, 41.f, 42.f, 43.f, 44.f,

            45.f, 46.f, 47.f, 48.f, 49.f,
            50.f, 51.f, 52.f, 53.f, 54.f,
            55.f, 56.f, 57.f, 58.f, 59.f,


            60.f, 61.f, 62.f, 63.f, 64.f,
            65.f, 66.f, 67.f, 68.f, 69.f,
            70.f, 71.f, 72.f, 73.f, 74.f,

            75.f, 76.f, 77.f, 78.f, 79.f,
            80.f, 81.f, 82.f, 83.f, 84.f,
            85.f, 86.f, 87.f, 88.f, 89.f
        },
        {
            37.f, 38.f, 39.f,
            42.f, 43.f, 44.f,

            67.f, 68.f, 69.f,
            72.f, 73.f, 74.f
        });
}

}
