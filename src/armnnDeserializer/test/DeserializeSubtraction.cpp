//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <string>

TEST_SUITE("Deserializer_Subtraction")
{
struct SubtractionFixture : public ParserFlatbuffersSerializeFixture
{
    explicit SubtractionFixture(const std::string & inputShape1,
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
                                   layerName: "inputLayer1",
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
                              },
                       }},
            },
            {
            layer_type: "InputLayer",
            layer: {
                   base: {
                         layerBindingId: 1,
                         base: {
                               index:1,
                               layerName: "inputLayer2",
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
                         },
                   }},
            },
            {
            layer_type: "SubtractionLayer",
            layer : {
                    base: {
                          index:2,
                          layerName: "subtractionLayer",
                          layerType: "Subtraction",
                          inputSlots: [{
                              index: 0,
                              connection: {sourceLayerIndex:0, outputSlotIndex:0 },
                          },
                          {
                              index: 1,
                              connection: {sourceLayerIndex:1, outputSlotIndex:0 },
                          }],
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
                               layerName: "outputLayer",
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

struct SimpleSubtractionFixture : SubtractionFixture
{
    SimpleSubtractionFixture() : SubtractionFixture("[ 1, 4 ]",
                                                    "[ 1, 4 ]",
                                                    "[ 1, 4 ]",
                                                    "QuantisedAsymm8") {}
};

struct SimpleSubtractionFixture2 : SubtractionFixture
{
    SimpleSubtractionFixture2() : SubtractionFixture("[ 1, 4 ]",
                                                     "[ 1, 4 ]",
                                                     "[ 1, 4 ]",
                                                     "Float32") {}
};

struct SimpleSubtractionFixtureBroadcast : SubtractionFixture
{
    SimpleSubtractionFixtureBroadcast() : SubtractionFixture("[ 1, 4 ]",
                                                             "[ 1, 1 ]",
                                                             "[ 1, 4 ]",
                                                             "Float32") {}
};

TEST_CASE_FIXTURE(SimpleSubtractionFixture, "SubtractionQuantisedAsymm8")
{
    RunTest<2, armnn::DataType::QAsymmU8>(
        0,
        {{"inputLayer1", { 4, 5, 6, 7 }},
         {"inputLayer2", { 3, 2, 1, 0 }}},
        {{"outputLayer", { 1, 3, 5, 7 }}});
}

TEST_CASE_FIXTURE(SimpleSubtractionFixture2, "SubtractionFloat32")
{
    RunTest<2, armnn::DataType::Float32>(
        0,
        {{"inputLayer1", { 4, 5, 6, 7 }},
         {"inputLayer2", { 3, 2, 1, 0 }}},
        {{"outputLayer", { 1, 3, 5, 7 }}});
}

TEST_CASE_FIXTURE(SimpleSubtractionFixtureBroadcast, "SubtractionBroadcast")
{
    RunTest<2, armnn::DataType::Float32>(
        0,
        {{"inputLayer1", { 4, 5, 6, 7 }},
         {"inputLayer2", { 2 }}},
        {{"outputLayer", { 2, 3, 4, 5 }}});
}

}
