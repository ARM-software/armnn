//
// Copyright © 2017 Arm Ltd. All rights reserved.
// SPDX-License-Identifier: MIT
//

#include "ParserFlatbuffersSerializeFixture.hpp"
#include <armnnDeserializer/IDeserializer.hpp>

#include <armnn/utility/IgnoreUnused.hpp>

#include <doctest/doctest.h>

#include <string>

TEST_SUITE("Deserializer_Add")
{
struct AddFixture : public ParserFlatbuffersSerializeFixture
{
    explicit AddFixture(const std::string & inputShape1,
                        const std::string & inputShape2,
                        const std::string & outputShape,
                        const std::string & dataType,
                        const std::string & activation="NONE")
    {
        armnn::IgnoreUnused(activation);
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
                layer_type: "AdditionLayer",
                layer : {
                        base: {
                             index:2,
                             layerName: "AdditionLayer",
                             layerType: "Addition",
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


struct SimpleAddFixture : AddFixture
{
    SimpleAddFixture() : AddFixture("[ 2, 2 ]",
                                    "[ 2, 2 ]",
                                    "[ 2, 2 ]",
                                    "QuantisedAsymm8") {}
};

struct SimpleAddFixture2 : AddFixture
{
    SimpleAddFixture2() : AddFixture("[ 2, 2, 1, 1 ]",
                                     "[ 2, 2, 1, 1 ]",
                                     "[ 2, 2, 1, 1 ]",
                                     "Float32") {}
};

TEST_CASE_FIXTURE(SimpleAddFixture, "AddQuantisedAsymm8")
{
  RunTest<2, armnn::DataType::QAsymmU8>(
      0,
      {{"InputLayer1", { 0, 1, 2, 3 }},
      {"InputLayer2", { 4, 5, 6, 7 }}},
      {{"OutputLayer", { 4, 6, 8, 10 }}});
}

TEST_CASE_FIXTURE(SimpleAddFixture2, "AddFloat32")
{
    RunTest<4, armnn::DataType::Float32>(
    0,
    {{"InputLayer1", { 111, 85, 226, 3 }},
    {"InputLayer2", {   5,   8,  10, 12 }}},
    {{"OutputLayer", { 116, 93, 236, 15 }}});
}

}
