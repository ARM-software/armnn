//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Division")
{
struct DivisionFixture : public ParserFlatbuffersSerializeFixture
{
    explicit DivisionFixture(const std::string & inputShape1,
                        const std::string & inputShape2,
                        const std::string & outputShape,
                        const std::string & dataType)
    {
        m_JsonString = R"(
        {
                inputIds: [0, 1],
                outputIds: [3],
                layers: [
                {
                    layer_type: "InputLayer",
                    layer: {
                          base: {
                                layerBindingId: 0,
                                base: {
                                    index: 0,
                                    layerName: "InputLayer1",
                                    layerType: "Input",
                                    inputSlots: [{
                                        index: 0,
                                        connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [ {
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + inputShape1 + R"(,
                                            dataType: )" + dataType + R"(
                                        },
                                    }],
                                 },}},
                },
                {
                layer_type: "InputLayer",
                layer: {
                       base: {
                            layerBindingId: 1,
                            base: {
                                  index:1,
                                  layerName: "InputLayer2",
                                  layerType: "Input",
                                  inputSlots: [{
                                      index: 0,
                                      connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                  }],
                                  outputSlots: [ {
                                      index: 0,
                                      tensorInfo: {
                                          dimensions: )" + inputShape2 + R"(,
                                          dataType: )" + dataType + R"(
                                      },
                                  }],
                                },}},
                },
                {
                layer_type: "DivisionLayer",
                layer : {
                        base: {
                             index:2,
                             layerName: "DivisionLayer",
                             layerType: "Division",
                             inputSlots: [
                                            {
                                             index: 0,
                                             connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                                            },
                                            {
                                             index: 1,
                                             connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                                            }
                             ],
                             outputSlots: [ {
                                 index: 0,
                                 tensorInfo: {
                                     dimensions: )" + outputShape + R"(,
                                     dataType: )" + dataType + R"(
                                 },
                             }],
                            }},
                },
                {
                layer_type: "OutputLayer",
                layer: {
                        base:{
                              layerBindingId: 0,
                              base: {
                                    index: 3,
                                    layerName: "OutputLayer",
                                    layerType: "Output",
                                    inputSlots: [{
                                        index: 0,
                                        connection: {sourceLayerIndex:2, outputSlotIndex:0 },
                                    }],
                                    outputSlots: [ {
                                        index: 0,
                                        tensorInfo: {
                                            dimensions: )" + outputShape + R"(,
                                            dataType: )" + dataType + R"(
                                        },
                                }],
                            }}},
                }]
         }
        )";
        Setup();
    }
};


struct SimpleDivisionFixture : DivisionFixture
{
    SimpleDivisionFixture() : DivisionFixture("[ 2, 2 ]",
                                              "[ 2, 2 ]",
                                              "[ 2, 2 ]",
                                              "QuantisedAsymm8") {}
};

struct SimpleDivisionFixture2 : DivisionFixture
{
    SimpleDivisionFixture2() : DivisionFixture("[ 2, 2, 1, 1 ]",
                                               "[ 2, 2, 1, 1 ]",
                                               "[ 2, 2, 1, 1 ]",
                                               "Float32") {}
};

TEST_CASE_FIXTURE(SimpleDivisionFixture, "DivisionQuantisedAsymm8")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
        0,
        {{"InputLayer1", { 0, 5, 24, 21 }},
         {"InputLayer2", { 4, 1, 6,  7 }}},
        {{"OutputLayer", { 0, 5, 3,  3 }}});
}

TEST_CASE_FIXTURE(SimpleDivisionFixture2, "DivisionFloat32")
{
    RunTest<4, armnn::DataType::Float32>(
        0,
        {{"InputLayer1", { 100, 40, 226, 9 }},
         {"InputLayer2", { 5,   8,  1,   3 }}},
        {{"OutputLayer", { 20,  5,  226, 3 }}});
}

}
