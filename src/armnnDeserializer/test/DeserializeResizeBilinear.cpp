//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include <boost/test/unit_test.hpp>
#include "ParserFlatbuffersSerializeFixture.hpp"
#include "../Deserializer.hpp"

#include <string>

BOOST_AUTO_TEST_SUITE(Deserializer)

struct ResizeBilinearFixture : public ParserFlatbuffersSerializeFixture
{
    explicit ResizeBilinearFixture(const std::string& inputShape,
                                   const std::string& targetWidth,
                                   const std::string& targetHeight,
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
                        layer_type: "ResizeBilinearLayer",
                        layer: {
                            base: {
                                index: 1,
                                layerName: "ResizeBilinearLayer",
                                layerType: "ResizeBilinear",
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
                                targetWidth: )" + targetWidth + R"(,
                                targetHeight: )" + targetHeight + R"(,
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

struct SimpleResizeBilinearFixture : ResizeBilinearFixture
{
    SimpleResizeBilinearFixture() : ResizeBilinearFixture("[1, 2, 2, 2]",
                                                          "1",
                                                          "1",
                                                          "NCHW",
                                                          "[1, 2, 1, 1]",
                                                          "Float32") {}
};

BOOST_FIXTURE_TEST_CASE(SimpleResizeBilinearFloat32, SimpleResizeBilinearFixture)
{
    RunTest<4, armnn::DataType::Float32>(0,
                                         {
                                              1.0f, 255.0f, 200.0f, 250.0f,
                                            250.0f, 200.0f, 250.0f,   1.0f
                                         },
                                         {
                                              1.0f, 250.0f
                                         });
}

BOOST_AUTO_TEST_SUITE_END()
