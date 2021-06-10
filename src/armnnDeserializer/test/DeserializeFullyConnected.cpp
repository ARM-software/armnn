//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("DeserializeParser_FullyConnected")
{
struct FullyConnectedFixture : public ParserFlatbuffersSerializeFixture
{
    explicit FullyConnectedFixture(const std::string & inputShape1,
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
                                    quantizationScale: 1.0,
                                    quantizationOffset: 0
                                    },
                                }]
                            },
                        }
                    },
                },
            {
            layer_type: "FullyConnectedLayer",
            layer : {
                base: {
                    index:1,
                    layerName: "FullyConnectedLayer",
                    layerType: "FullyConnected",
                    inputSlots: [{
                            index: 0,
                            connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                        }],
                    outputSlots: [{
                        index: 0,
                        tensorInfo: {
                            dimensions: )" + outputShape + R"(,
                            dataType: )" + dataType + R"(,
                            quantizationScale: 2.0,
                            quantizationOffset: 0
                        },
                        }],
                    },
                descriptor: {
                    biasEnabled: false,
                    transposeWeightsMatrix: true
                    },
                weights: {
                    info: {
                             dimensions: )" + weightsShape + R"(,
                             dataType: )" + dataType + R"(,
                             quantizationScale: 1.0,
                             quantizationOffset: 0
                         },
                    data_type: ByteData,
                    data: {
                        data: [
                            2, 3, 4, 5
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

struct FullyConnectedWithNoBiasFixture : FullyConnectedFixture
{
    FullyConnectedWithNoBiasFixture()
        : FullyConnectedFixture("[ 1, 4, 1, 1 ]",     // inputShape
                                "[ 1, 1 ]",           // outputShape
                                "[ 1, 4 ]",           // filterShape
                                "QuantisedAsymm8")     // filterData
    {}
};

TEST_CASE_FIXTURE(FullyConnectedWithNoBiasFixture, "FullyConnectedWithNoBias")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
         0,
         {{"InputLayer",  { 10, 20, 30, 40 }}},
         {{"OutputLayer", { 400/2 }}});
}

}
