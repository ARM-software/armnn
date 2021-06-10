//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_StridedSlice")
{
struct StridedSliceFixture : public ParserFlatbuffersSerializeFixture
{
    explicit StridedSliceFixture(const std::string& inputShape,
                                 const std::string& begin,
                                 const std::string& end,
                                 const std::string& stride,
                                 const std::string& beginMask,
                                 const std::string& endMask,
                                 const std::string& shrinkAxisMask,
                                 const std::string& ellipsisMask,
                                 const std::string& newAxisMask,
                                 const std::string& dataLayout,
                                 const std::string& outputShape,
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
                        layer_type: "StridedSliceLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "StridedSliceLayer",
                                layerType: "StridedSlice",
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
                                end: )" + end + R"(,
                                stride: )" + stride + R"(,
                                beginMask: )" + beginMask + R"(,
                                endMask: )" + endMask + R"(,
                                shrinkAxisMask: )" + shrinkAxisMask + R"(,
                                ellipsisMask: )" + ellipsisMask + R"(,
                                newAxisMask: )" + newAxisMask + R"(,
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

struct SimpleStridedSliceFixture : StridedSliceFixture
{
    SimpleStridedSliceFixture() : StridedSliceFixture("[ 3, 2, 3, 1 ]",
                                                      "[ 0, 0, 0, 0 ]",
                                                      "[ 3, 2, 3, 1 ]",
                                                      "[ 2, 2, 2, 1 ]",
                                                      "0",
                                                      "0",
                                                      "0",
                                                      "0",
                                                      "0",
                                                      "NCHW",
                                                      "[ 2, 1, 2, 1 ]",
                                                      "Float32") {}
};

TEST_CASE_FIXTURE(SimpleStridedSliceFixture, "SimpleStridedSliceFloat32")
{
    RunTest<4, armnn::DataType::Float32>(0,
                                         {
                                             1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                             3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                             5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
                                         },
                                         {
                                             1.0f, 1.0f, 5.0f, 5.0f
                                         });
}

struct StridedSliceMaskFixture : StridedSliceFixture
{
    StridedSliceMaskFixture() : StridedSliceFixture("[ 3, 2, 3, 1 ]",
                                                    "[ 1, 1, 1, 1 ]",
                                                    "[ 1, 1, 1, 1 ]",
                                                    "[ 1, 1, 1, 1 ]",
                                                    "15",
                                                    "15",
                                                    "0",
                                                    "0",
                                                    "0",
                                                    "NCHW",
                                                    "[ 3, 2, 3, 1 ]",
                                                    "Float32") {}
};

TEST_CASE_FIXTURE(StridedSliceMaskFixture, "StridedSliceMaskFloat32")
{
    RunTest<4, armnn::DataType::Float32>(0,
                                         {
                                             1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                             3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                             5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
                                         },
                                         {
                                             1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f,
                                             3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f,
                                             5.0f, 5.0f, 5.0f, 6.0f, 6.0f, 6.0f
                                         });
}

}
