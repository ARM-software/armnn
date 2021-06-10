//
// Copyright © 2019 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"

#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_DepthToSpace")
{
struct DepthToSpaceFixture : public ParserFlatbuffersSerializeFixture
{
    explicit DepthToSpaceFixture(const std::string& inputShape,
                                 const std::string& outputShape,
                                 const std::string& blockSize,
                                 const std::string& dataLayout,
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
                        layer_type: "DepthToSpaceLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "DepthToSpaceLayer",
                                layerType: "DepthToSpace",
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
                                blockSize: )" + blockSize + R"(,
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

struct DepthToSpaceFloat32Fixture : DepthToSpaceFixture
{
    DepthToSpaceFloat32Fixture() : DepthToSpaceFixture("[ 1, 2, 2, 4 ]", // input shape
                                                       "[ 1, 4, 4, 1 ]", // output shape
                                                       "2",              // block size
                                                       "NHWC",           // data layout
                                                       "Float32") {}     // data type
};

TEST_CASE_FIXTURE(DepthToSpaceFloat32Fixture, "DepthToSpaceFloat32")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        {
             1.f,  2.f,  3.f,  4.f,
             5.f,  6.f,  7.f,  8.f,
             9.f, 10.f, 11.f, 12.f,
            13.f, 14.f, 15.f, 16.f
        },
        {
             1.f,  2.f,  5.f,  6.f,
             3.f,  4.f,  7.f,  8.f,
             9.f, 10.f, 13.f, 14.f,
            11.f, 12.f, 15.f, 16.f
        });
}

}
