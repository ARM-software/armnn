//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_BatchNormalization")
{
struct BatchNormalizationFixture : public ParserFlatbuffersSerializeFixture
{
    explicit BatchNormalizationFixture(const std::string &inputShape,
                                       const std::string &outputShape,
                                       const std::string &meanShape,
                                       const std::string &varianceShape,
                                       const std::string &offsetShape,
                                       const std::string &scaleShape,
                                       const std::string &dataType,
                                       const std::string &dataLayout)
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
        layer_type: "BatchNormalizationLayer",
        layer : {
            base: {
                index:1,
                layerName: "BatchNormalizationLayer",
                layerType: "BatchNormalization",
                inputSlots: [{
                        index: 0,
                        connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                   }],
                outputSlots: [{
                    index: 0,
                    tensorInfo: {
                        dimensions: )" + outputShape + R"(,
                        dataType: ")" + dataType + R"("
                    },
                    }],
                },
            descriptor: {
                eps: 0.0010000000475,
                dataLayout: ")" + dataLayout + R"("
                },
            mean: {
                info: {
                         dimensions: )" + meanShape + R"(,
                         dataType: ")" + dataType + R"("
                     },
                data_type: IntData,
                data: {
                    data: [1084227584],
                    }
                },
            variance: {
                info: {
                         dimensions: )" + varianceShape + R"(,
                         dataType: ")" + dataType + R"("
                     },
               data_type: IntData,
                data: {
                    data: [1073741824],
                    }
                },
            beta: {
                info: {
                         dimensions: )" + offsetShape + R"(,
                         dataType: ")" + dataType + R"("
                     },
                data_type: IntData,
                data: {
                    data: [0],
                    }
                },
            gamma: {
                info: {
                         dimensions: )" + scaleShape + R"(,
                         dataType: ")" + dataType + R"("
                     },
                data_type: IntData,
                data: {
                    data: [1065353216],
                    }
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
                        connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                    }],
                    outputSlots: [ {
                        index: 0,
                        tensorInfo: {
                            dimensions: )" + outputShape + R"(,
                            dataType: ")" + dataType + R"("
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

struct BatchNormFixture : BatchNormalizationFixture
{
    BatchNormFixture():BatchNormalizationFixture("[ 1, 3, 3, 1 ]",
                                                 "[ 1, 3, 3, 1 ]",
                                                 "[ 1 ]",
                                                 "[ 1 ]",
                                                 "[ 1 ]",
                                                 "[ 1 ]",
                                                 "Float32",
                                                 "NHWC"){}
};

TEST_CASE_FIXTURE(BatchNormFixture, "BatchNormalizationFloat32")
{
    RunTest<4, armnn::DataType::Float32>(0,
                                         {{"InputLayer", { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f }}},
                                         {{"OutputLayer",{ -2.8277204f, -2.12079024f, -1.4138602f,
                                           -0.7069301f,  0.0f,         0.7069301f,
                                           1.4138602f,  2.12079024f,  2.8277204f }}});
}

}
