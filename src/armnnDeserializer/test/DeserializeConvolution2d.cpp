//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Convolution2D")
{
struct Convolution2dFixture : public ParserFlatbuffersSerializeFixture
{
    explicit Convolution2dFixture(const std::string & inputShape1,
                                  const std::string & outputShape,
                                  const std::string & weightsShape,
                                  const std::string & dataType)
    {
        m_JsonString = R"(
        {
            inputIds: [0],
            outputIds: [2],
            layers: [{
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
                                    dimensions: )" + inputShape1 + R"(,
                                    dataType: )" + dataType + R"(,
                                    quantizationScale: 0.5,
                                    quantizationOffset: 0
                                    },
                                }]
                            },
                        }
                    },
                },
            {
            layer_type: "Convolution2dLayer",
            layer : {
                base: {
                    index:1,
                    layerName: "Convolution2dLayer",
                    layerType: "Convolution2d",
                    inputSlots: [{
                            index: 0,
                            connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                        }],
                    outputSlots: [{
                        index: 0,
                        tensorInfo: {
                            dimensions: )" + outputShape + R"(,
                            dataType: )" + dataType + R"(
                        },
                        }],
                    },
                descriptor: {
                    padLeft: 1,
                    padRight: 1,
                    padTop: 1,
                    padBottom: 1,
                    strideX: 2,
                    strideY: 2,
                    biasEnabled: false,
                    dataLayout: NHWC
                    },
                weights: {
                    info: {
                             dimensions: )" + weightsShape + R"(,
                             dataType: )" + dataType + R"(
                         },
                    data_type: IntData,
                    data: {
                        data: [
                            1082130432, 1084227584, 1086324736,
                            0 ,0 ,0 ,
                            1077936128, 1073741824, 1065353216
                            ],
                        }
                    }
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
                            connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                        }],
                        outputSlots: [ {
                            index: 0,
                            tensorInfo: {
                                dimensions: )" + outputShape + R"(,
                                dataType: )" + dataType + R"(
                            },
                        }],
                    }
                }},
            }]
        }
        )";
        Setup();
    }
};

struct SimpleConvolution2dFixture : Convolution2dFixture
{
    SimpleConvolution2dFixture() : Convolution2dFixture("[ 1, 5, 5, 1 ]",
                                     "[ 1, 3, 3, 1 ]",
                                     "[ 1, 3, 3, 1 ]",
                                     "Float32") {}
};

TEST_CASE_FIXTURE(SimpleConvolution2dFixture, "Convolution2dFloat32")
{
    RunTest<4, armnn::DataType::Float32>(
            0,
            {{"InputLayer", {1, 5, 2, 3, 5, 8, 7, 3, 6, 3, 3, 3, 9, 1, 9, 4, 1, 8, 1, 3, 6, 8, 1, 9, 2}}},
            {{"OutputLayer", {23, 33, 24, 91, 99, 48, 26, 50, 19}}});
}

}
